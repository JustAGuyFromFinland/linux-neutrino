// SPDX-License-Identifier: GPL-2.0
/*
 * Per Entity Load Tracking (PELT)
 *
 *  Copyright (C) 2007 Red Hat, Inc., Ingo Molnar <mingo@redhat.com>
 *
 *  Interactivity improvements by Mike Galbraith
 *  (C) 2007 Mike Galbraith <efault@gmx.de>
 *
 *  Various enhancements by Dmitry Adamushko.
 *  (C) 2007 Dmitry Adamushko <dmitry.adamushko@gmail.com>
 *
 *  Group scheduling enhancements by Srivatsa Vaddagiri
 *  Copyright IBM Corporation, 2007
 *  Author: Srivatsa Vaddagiri <vatsa@linux.vnet.ibm.com>
 *
 *  Scaled math optimizations by Thomas Gleixner
 *  Copyright (C) 2007, Linutronix GmbH, Thomas Gleixner <tglx@kernel.org>
 *
 *  Adaptive scheduling granularity, math enhancements by Peter Zijlstra
 *  Copyright (C) 2007 Red Hat, Inc., Peter Zijlstra
 *
 *  Move PELT related code from fair.c into this pelt.c file
 *  Author: Vincent Guittot <vincent.guittot@linaro.org>
 */
#include "pelt.h"
#include "pelt-simd.h"

/*
 * Approximate:
 *   val * y^n,    where y^32 ~= 0.5 (~1 scheduling period)
 *
 * Optimized branchless version to reduce branch mispredictions.
 * The original code had conditional branches that cause pipeline stalls
 * on modern CPUs with deep pipelines.
 */
static __always_inline u64 decay_load(u64 val, u64 n)
{
	u64 periods, remainder, shifted_val, result;
	u64 zero_mask, large_n_mask;

	/*
	 * Branchless bounds check: if n > 2016, result should be 0
	 * We compute this as a mask: all 1s if should zero, all 0s otherwise
	 */
	zero_mask = (s64)(n - (LOAD_AVG_PERIOD * 63 + 1)) >> 63;
	zero_mask = ~zero_mask; /* Invert: all 1s if n > 2016 */

	/*
	 * As y^PERIOD = 1/2, we can combine:
	 *    y^n = 1/2^(n/PERIOD) * y^(n%PERIOD)
	 * With a look-up table which covers y^n (n<PERIOD)
	 *
	 * Compute periods and remainder without branches
	 */
	periods = n >> 5; /* n / LOAD_AVG_PERIOD (LOAD_AVG_PERIOD = 32) */
	remainder = n & 31; /* n % LOAD_AVG_PERIOD */

	/*
	 * Clamp periods to 63 to avoid undefined shift behavior
	 * Using branchless min: min(periods, 63)
	 */
	large_n_mask = (s64)(periods - 64) >> 63; /* All 1s if periods < 64 */
	periods = (periods & large_n_mask) | (63 & ~large_n_mask);

	/* Shift val by periods (equivalent to dividing by 2^periods) */
	shifted_val = val >> periods;

	/* Apply y^remainder decay using lookup table */
	result = mul_u64_u32_shr(shifted_val, runnable_avg_yN_inv[remainder], 32);

	/* Return 0 if zero_mask is set (n was too large) */
	return result & ~zero_mask;
}

static u32 __accumulate_pelt_segments(u64 periods, u32 d1, u32 d3)
{
	u32 c1, c2, c3 = d3; /* y^0 == 1 */

	/*
	 * c1 = d1 y^p
	 */
	c1 = decay_load((u64)d1, periods);

	/*
	 *            p-1
	 * c2 = 1024 \Sum y^n
	 *            n=1
	 *
	 *              inf        inf
	 *    = 1024 ( \Sum y^n - \Sum y^n - y^0 )
	 *              n=0        n=p
	 */
	c2 = LOAD_AVG_MAX - decay_load(LOAD_AVG_MAX, periods) - 1024;

	return c1 + c2 + c3;
}

/*
 * Accumulate the three separate parts of the sum; d1 the remainder
 * of the last (incomplete) period, d2 the span of full periods and d3
 * the remainder of the (incomplete) current period.
 *
 *           d1          d2           d3
 *           ^           ^            ^
 *           |           |            |
 *         |<->|<----------------->|<--->|
 * ... |---x---|------| ... |------|-----x (now)
 *
 *                           p-1
 * u' = (u + d1) y^p + 1024 \Sum y^n + d3 y^0
 *                           n=1
 *
 *    = u y^p +					(Step 1)
 *
 *                     p-1
 *      d1 y^p + 1024 \Sum y^n + d3 y^0		(Step 2)
 *                     n=1
 *
 * Optimized branchless version: Instead of multiple if statements,
 * we compute all paths and use conditional arithmetic to select results.
 * This eliminates branch mispredictions in this hot path.
 */
static __always_inline u32
accumulate_sum(u64 delta, struct sched_avg *sa,
	       unsigned long load, unsigned long runnable, int running)
{
	u32 contrib = (u32)delta; /* p == 0 -> delta < 1024 */
	u64 periods;
	u64 periods_mask, load_mask, runnable_mask, running_mask;
	u64 old_load_sum, old_runnable_sum, old_util_sum;
	u64 new_load_sum, new_runnable_sum, new_util_sum;
	u32 old_period_contrib;

	delta += sa->period_contrib;
	periods = delta >> 10; /* delta / 1024 - A period is 1024us (~1ms) */

	/*
	 * Create branchless masks for conditional operations
	 * periods_mask: all 1s if periods != 0, all 0s otherwise
	 */
	periods_mask = (u64)(-(s64)!!periods);
	load_mask = (u64)(-(s64)!!load);
	runnable_mask = (u64)(-(s64)!!runnable);
	running_mask = (u64)(-(s64)!!running);

	/* Save old values for conditional update */
	old_load_sum = sa->load_sum;
	old_runnable_sum = sa->runnable_sum;
	old_util_sum = sa->util_sum;
	old_period_contrib = sa->period_contrib;

	/*
	 * Step 1: decay old *_sum if we crossed period boundaries.
	 * Compute decayed values (will be selected if periods != 0)
	 */
	new_load_sum = decay_load(old_load_sum, periods);
	new_runnable_sum = decay_load(old_runnable_sum, periods);
	new_util_sum = decay_load(old_util_sum, periods);

	/* Branchless select: use decayed values if periods != 0 */
	sa->load_sum = (new_load_sum & periods_mask) | (old_load_sum & ~periods_mask);
	sa->runnable_sum = (new_runnable_sum & periods_mask) | (old_runnable_sum & ~periods_mask);
	sa->util_sum = (u32)((new_util_sum & periods_mask) | (old_util_sum & ~periods_mask));

	/*
	 * Step 2: Compute contribution (only meaningful if periods != 0 && load != 0)
	 * We always compute it but only use it when both conditions are true
	 */
	{
		u64 new_delta = delta & 1023; /* delta % 1024 */
		u32 new_contrib;
		u64 contrib_mask;

		/* Compute contrib for the periods case */
		new_contrib = __accumulate_pelt_segments(periods,
				1024 - old_period_contrib, (u32)new_delta);

		/* Use new_contrib if periods && load, otherwise use original delta */
		contrib_mask = periods_mask & load_mask;
		contrib = (u32)((new_contrib & contrib_mask) | (contrib & ~contrib_mask));

		/* Update period_contrib: new_delta if periods, else delta */
		sa->period_contrib = (u32)((new_delta & periods_mask) | (delta & ~periods_mask));
	}

	/*
	 * Accumulate contributions using branchless arithmetic
	 * The mask ensures we only add when the condition is true
	 */
	sa->load_sum += (load * contrib) & load_mask;
	sa->runnable_sum += ((runnable * contrib) << SCHED_CAPACITY_SHIFT) & runnable_mask;
	sa->util_sum += (u32)(((u64)contrib << SCHED_CAPACITY_SHIFT) & running_mask);

	return periods;
}

/*
 * We can represent the historical contribution to runnable average as the
 * coefficients of a geometric series.  To do this we sub-divide our runnable
 * history into segments of approximately 1ms (1024us); label the segment that
 * occurred N-ms ago p_N, with p_0 corresponding to the current period, e.g.
 *
 * [<- 1024us ->|<- 1024us ->|<- 1024us ->| ...
 *      p0            p1           p2
 *     (now)       (~1ms ago)  (~2ms ago)
 *
 * Let u_i denote the fraction of p_i that the entity was runnable.
 *
 * We then designate the fractions u_i as our co-efficients, yielding the
 * following representation of historical load:
 *   u_0 + u_1*y + u_2*y^2 + u_3*y^3 + ...
 *
 * We choose y based on the with of a reasonably scheduling period, fixing:
 *   y^32 = 0.5
 *
 * This means that the contribution to load ~32ms ago (u_32) will be weighted
 * approximately half as much as the contribution to load within the last ms
 * (u_0).
 *
 * When a period "rolls over" and we have new u_0`, multiplying the previous
 * sum again by y is sufficient to update:
 *   load_avg = u_0` + y*(u_0 + u_1*y + u_2*y^2 + ... )
 *            = u_0 + u_1*y + u_2*y^2 + ... [re-labeling u_i --> u_{i+1}]
 *
 * Optimized with branchless operations for better pipeline utilization
 * on modern superscalar CPUs with deep pipelines (Haswell+).
 */
static __always_inline int
___update_load_sum(u64 now, struct sched_avg *sa,
		  unsigned long load, unsigned long runnable, int running)
{
	u64 delta;
	s64 signed_delta;
	u64 negative_mask;
	u64 load_mask;
	u32 periods;

	delta = now - sa->last_update_time;
	signed_delta = (s64)delta;

	/*
	 * Handle time going backwards (during sched clock init / TSC swap)
	 * using branchless conditional update.
	 *
	 * negative_mask is all 1s if delta is negative, all 0s otherwise
	 */
	negative_mask = (u64)(signed_delta >> 63);

	/* Update last_update_time unconditionally if delta is negative */
	sa->last_update_time = (now & negative_mask) |
			       (sa->last_update_time & ~negative_mask);

	/* Early return for negative delta - this is rare, keep the branch */
	if (unlikely(negative_mask))
		return 0;

	/*
	 * Use 1024ns as the unit of measurement since it's a reasonable
	 * approximation of 1us and fast to compute.
	 */
	delta >>= 10;

	/* Zero delta means no update needed */
	if (!delta)
		return 0;

	sa->last_update_time += delta << 10;

	/*
	 * running is a subset of runnable (weight) so running can't be set if
	 * runnable is clear. But there are some corner cases where the current
	 * se has been already dequeued but cfs_rq->curr still points to it.
	 * This means that weight will be 0 but not running for a sched_entity
	 * but also for a cfs_rq if the latter becomes idle.
	 *
	 * Branchless: if load is 0, force runnable and running to 0
	 */
	load_mask = (u64)(-(s64)!!load);
	runnable &= load_mask;
	running &= (int)load_mask;

	/*
	 * Now we know we crossed measurement unit boundaries. The *_avg
	 * accrues by two steps:
	 *
	 * Step 1: accumulate *_sum since last_update_time. If we haven't
	 * crossed period boundaries, finish.
	 */
	periods = accumulate_sum(delta, sa, load, runnable, running);

	/* Return 1 if we crossed period boundaries, 0 otherwise */
	return !!periods;
}

/*
 * When syncing *_avg with *_sum, we must take into account the current
 * position in the PELT segment otherwise the remaining part of the segment
 * will be considered as idle time whereas it's not yet elapsed and this will
 * generate unwanted oscillation in the range [1002..1024[.
 *
 * The max value of *_sum varies with the position in the time segment and is
 * equals to :
 *
 *   LOAD_AVG_MAX*y + sa->period_contrib
 *
 * which can be simplified into:
 *
 *   LOAD_AVG_MAX - 1024 + sa->period_contrib
 *
 * because LOAD_AVG_MAX*y == LOAD_AVG_MAX-1024
 *
 * The same care must be taken when a sched entity is added, updated or
 * removed from a cfs_rq and we need to update sched_avg. Scheduler entities
 * and the cfs rq, to which they are attached, have the same position in the
 * time segment because they use the same clock. This means that we can use
 * the period_contrib of cfs_rq when updating the sched_avg of a sched_entity
 * if it's more convenient.
 */
static __always_inline void
___update_load_avg(struct sched_avg *sa, unsigned long load)
{
	u32 divider = get_pelt_divider(sa);

	/*
	 * Step 2: update *_avg.
	 */
	sa->load_avg = div_u64(load * sa->load_sum, divider);
	sa->runnable_avg = div_u64(sa->runnable_sum, divider);
	WRITE_ONCE(sa->util_avg, sa->util_sum / divider);
}

/*
 * sched_entity:
 *
 *   task:
 *     se_weight()   = se->load.weight
 *     se_runnable() = !!on_rq
 *
 *   group: [ see update_cfs_group() ]
 *     se_weight()   = tg->weight * grq->load_avg / tg->load_avg
 *     se_runnable() = grq->h_nr_runnable
 *
 *   runnable_sum = se_runnable() * runnable = grq->runnable_sum
 *   runnable_avg = runnable_sum
 *
 *   load_sum := runnable
 *   load_avg = se_weight(se) * load_sum
 *
 * cfq_rq:
 *
 *   runnable_sum = \Sum se->avg.runnable_sum
 *   runnable_avg = \Sum se->avg.runnable_avg
 *
 *   load_sum = \Sum se_weight(se) * se->avg.load_sum
 *   load_avg = \Sum se->avg.load_avg
 */

int __update_load_avg_blocked_se(u64 now, struct sched_entity *se)
{
	if (___update_load_sum(now, &se->avg, 0, 0, 0)) {
		___update_load_avg(&se->avg, se_weight(se));
		trace_pelt_se_tp(se);
		return 1;
	}

	return 0;
}

int __update_load_avg_se(u64 now, struct cfs_rq *cfs_rq, struct sched_entity *se)
{
	if (___update_load_sum(now, &se->avg, !!se->on_rq, se_runnable(se),
				cfs_rq->curr == se)) {

		___update_load_avg(&se->avg, se_weight(se));
		cfs_se_util_change(&se->avg);
		trace_pelt_se_tp(se);
		return 1;
	}

	return 0;
}

int __update_load_avg_cfs_rq(u64 now, struct cfs_rq *cfs_rq)
{
	if (___update_load_sum(now, &cfs_rq->avg,
				scale_load_down(cfs_rq->load.weight),
				cfs_rq->h_nr_runnable,
				cfs_rq->curr != NULL)) {

		___update_load_avg(&cfs_rq->avg, 1);
		trace_pelt_cfs_tp(cfs_rq);
		return 1;
	}

	return 0;
}

/*
 * rt_rq:
 *
 *   util_sum = \Sum se->avg.util_sum but se->avg.util_sum is not tracked
 *   util_sum = cpu_scale * load_sum
 *   runnable_sum = util_sum
 *
 *   load_avg and runnable_avg are not supported and meaningless.
 *
 */

int update_rt_rq_load_avg(u64 now, struct rq *rq, int running)
{
	if (___update_load_sum(now, &rq->avg_rt,
				running,
				running,
				running)) {

		___update_load_avg(&rq->avg_rt, 1);
		trace_pelt_rt_tp(rq);
		return 1;
	}

	return 0;
}

/*
 * dl_rq:
 *
 *   util_sum = \Sum se->avg.util_sum but se->avg.util_sum is not tracked
 *   util_sum = cpu_scale * load_sum
 *   runnable_sum = util_sum
 *
 *   load_avg and runnable_avg are not supported and meaningless.
 *
 */

int update_dl_rq_load_avg(u64 now, struct rq *rq, int running)
{
	if (___update_load_sum(now, &rq->avg_dl,
				running,
				running,
				running)) {

		___update_load_avg(&rq->avg_dl, 1);
		trace_pelt_dl_tp(rq);
		return 1;
	}

	return 0;
}

#ifdef CONFIG_SCHED_HW_PRESSURE
/*
 * hardware:
 *
 *   load_sum = \Sum se->avg.load_sum but se->avg.load_sum is not tracked
 *
 *   util_avg and runnable_load_avg are not supported and meaningless.
 *
 * Unlike rt/dl utilization tracking that track time spent by a cpu
 * running a rt/dl task through util_avg, the average HW pressure is
 * tracked through load_avg. This is because HW pressure signal is
 * time weighted "delta" capacity unlike util_avg which is binary.
 * "delta capacity" =  actual capacity  -
 *			capped capacity a cpu due to a HW event.
 */

int update_hw_load_avg(u64 now, struct rq *rq, u64 capacity)
{
	if (___update_load_sum(now, &rq->avg_hw,
			       capacity,
			       capacity,
			       capacity)) {
		___update_load_avg(&rq->avg_hw, 1);
		trace_pelt_hw_tp(rq);
		return 1;
	}

	return 0;
}
#endif /* CONFIG_SCHED_HW_PRESSURE */

#ifdef CONFIG_HAVE_SCHED_AVG_IRQ
/*
 * IRQ:
 *
 *   util_sum = \Sum se->avg.util_sum but se->avg.util_sum is not tracked
 *   util_sum = cpu_scale * load_sum
 *   runnable_sum = util_sum
 *
 *   load_avg and runnable_avg are not supported and meaningless.
 *
 */

int update_irq_load_avg(struct rq *rq, u64 running)
{
	int ret = 0;

	/*
	 * We can't use clock_pelt because IRQ time is not accounted in
	 * clock_task. Instead we directly scale the running time to
	 * reflect the real amount of computation
	 */
	running = cap_scale(running, arch_scale_freq_capacity(cpu_of(rq)));
	running = cap_scale(running, arch_scale_cpu_capacity(cpu_of(rq)));

	/*
	 * We know the time that has been used by interrupt since last update
	 * but we don't when. Let be pessimistic and assume that interrupt has
	 * happened just before the update. This is not so far from reality
	 * because interrupt will most probably wake up task and trig an update
	 * of rq clock during which the metric is updated.
	 * We start to decay with normal context time and then we add the
	 * interrupt context time.
	 * We can safely remove running from rq->clock because
	 * rq->clock += delta with delta >= running
	 */
	ret = ___update_load_sum(rq->clock - running, &rq->avg_irq,
				0,
				0,
				0);
	ret += ___update_load_sum(rq->clock, &rq->avg_irq,
				1,
				1,
				1);

	if (ret) {
		___update_load_avg(&rq->avg_irq, 1);
		trace_pelt_irq_tp(rq);
	}

	return ret;
}
#endif /* CONFIG_HAVE_SCHED_AVG_IRQ */

/*
 * Load avg and utiliztion metrics need to be updated periodically and before
 * consumption. This function updates the metrics for all subsystems except for
 * the fair class. @rq must be locked and have its clock updated.
 */
bool update_other_load_avgs(struct rq *rq)
{
	u64 now = rq_clock_pelt(rq);
	const struct sched_class *curr_class = rq->donor->sched_class;
	unsigned long hw_pressure = arch_scale_hw_pressure(cpu_of(rq));

	lockdep_assert_rq_held(rq);

	/* hw_pressure doesn't care about invariance */
	return update_rt_rq_load_avg(now, rq, curr_class == &rt_sched_class) |
		update_dl_rq_load_avg(now, rq, curr_class == &dl_sched_class) |
		update_hw_load_avg(rq_clock_task(rq), rq, hw_pressure) |
		update_irq_load_avg(rq, 0);
}

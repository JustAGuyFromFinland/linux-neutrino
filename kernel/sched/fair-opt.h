/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Cache optimization helpers for CFS scheduler
 *
 * This header provides prefetching and branchless helpers specifically
 * optimized for the CFS load balancing hot paths to reduce cache misses.
 *
 * Copyright (C) 2024
 */
#ifndef _KERNEL_SCHED_FAIR_OPT_H
#define _KERNEL_SCHED_FAIR_OPT_H

#include <linux/compiler.h>
#include <linux/prefetch.h>

/*
 * Prefetch distance for load balancing loops.
 * This determines how many CPUs ahead we prefetch rq data.
 * Tuned for typical L2 cache latency (~12 cycles) and loop body (~50 cycles).
 */
#define LB_PREFETCH_DISTANCE 2

/*
 * Prefetch a runqueue structure for upcoming load balancing access.
 * The rq structure is large, so we prefetch the most commonly accessed
 * fields in the load balancing path.
 */
static __always_inline void prefetch_rq_lb(struct rq *rq)
{
	if (likely(rq)) {
		/* Prefetch nr_running and cfs.h_nr_runnable (hot fields) */
		prefetch(&rq->nr_running);
		/* Prefetch the cfs runqueue structure */
		prefetch(&rq->cfs);
		/* Prefetch cpu_capacity which is accessed in load calculations */
		prefetch(&rq->cpu_capacity);
	}
}

/*
 * Prefetch-ahead helper for for_each_cpu loops in load balancing.
 * Call this at the start of each iteration with the next CPU(s).
 */
static __always_inline void lb_prefetch_next_cpus(const struct cpumask *mask,
						  int current_cpu,
						  int distance)
{
	int next_cpu = current_cpu;
	int i;

	for (i = 0; i < distance; i++) {
		next_cpu = cpumask_next(next_cpu, mask);
		if (next_cpu < nr_cpu_ids)
			prefetch_rq_lb(cpu_rq(next_cpu));
		else
			break;
	}
}

/*
 * Branchless minimum for unsigned long (used in capacity calculations)
 */
static __always_inline unsigned long branchless_min_ul(unsigned long a,
						       unsigned long b)
{
	unsigned long diff = a - b;
	unsigned long mask = (long)diff >> (BITS_PER_LONG - 1);
	return (a & mask) | (b & ~mask);
}

/*
 * Branchless maximum for unsigned long (used in capacity calculations)
 */
static __always_inline unsigned long branchless_max_ul(unsigned long a,
						       unsigned long b)
{
	unsigned long diff = a - b;
	unsigned long mask = (long)diff >> (BITS_PER_LONG - 1);
	return (b & mask) | (a & ~mask);
}

/*
 * Branchless conditional add: adds value to sum only if condition is true
 * This is useful for accumulating statistics in loops
 */
static __always_inline unsigned long branchless_add_if(unsigned long sum,
						       unsigned long value,
						       int condition)
{
	unsigned long mask = (unsigned long)(-(long)!!condition);
	return sum + (value & mask);
}

/*
 * Branchless conditional increment
 */
static __always_inline unsigned int branchless_inc_if(unsigned int count,
						      int condition)
{
	return count + !!condition;
}

/*
 * SoA (Structure of Arrays) for batch processing CPU statistics
 * in load balancing. Instead of accessing rq structures one by one,
 * we batch-gather statistics for better cache utilization.
 */
#define LB_BATCH_SIZE 8

struct lb_cpu_stats_soa {
	unsigned long loads[LB_BATCH_SIZE] ____cacheline_aligned_in_smp;
	unsigned long utils[LB_BATCH_SIZE];
	unsigned long runnables[LB_BATCH_SIZE];
	unsigned int nr_running[LB_BATCH_SIZE];
	unsigned int h_nr_runnable[LB_BATCH_SIZE];
	unsigned int idle_flags[LB_BATCH_SIZE]; /* 1 if idle, 0 otherwise */
	int cpus[LB_BATCH_SIZE];
	int count;
} ____cacheline_aligned_in_smp;

/*
 * Initialize lb_cpu_stats_soa structure.
 * The actual gathering is done inline in fair.c where cpu_load/cpu_runnable
 * are available.
 */
static inline void init_cpu_stats_batch(struct lb_cpu_stats_soa *stats)
{
	stats->count = 0;
}

/*
 * Process batched CPU statistics with SIMD-friendly accumulation.
 * Designed to be vectorizable by the compiler with -march=haswell.
 */
static inline void accumulate_batch_stats(struct lb_cpu_stats_soa *stats,
					  unsigned long *total_load,
					  unsigned long *total_util,
					  unsigned long *total_runnable,
					  unsigned int *total_nr_running,
					  unsigned int *total_h_nr_runnable,
					  unsigned int *idle_count)
{
	int i;
	unsigned long load_acc = 0, util_acc = 0, run_acc = 0;
	unsigned int nr_acc = 0, h_nr_acc = 0, idle_acc = 0;

	/*
	 * This loop is designed to be auto-vectorized by GCC/Clang
	 * with -march=haswell -mavx2. The separate accumulators avoid
	 * loop-carried dependencies that would prevent vectorization.
	 */
	for (i = 0; i < stats->count; i++) {
		load_acc += stats->loads[i];
		util_acc += stats->utils[i];
		run_acc += stats->runnables[i];
		nr_acc += stats->nr_running[i];
		h_nr_acc += stats->h_nr_runnable[i];
		idle_acc += stats->idle_flags[i];
	}

	*total_load += load_acc;
	*total_util += util_acc;
	*total_runnable += run_acc;
	*total_nr_running += nr_acc;
	*total_h_nr_runnable += h_nr_acc;
	*idle_count += idle_acc;
}

/*
 * Optimized version of group capacity calculation using branchless ops.
 * Processes capacity values without conditional branches.
 */
static inline void update_capacity_bounds_branchless(unsigned long capacity,
						     unsigned long *min_cap,
						     unsigned long *max_cap)
{
	*min_cap = branchless_min_ul(capacity, *min_cap);
	*max_cap = branchless_max_ul(capacity, *max_cap);
}

/*
 * Memory barrier hint for load balancing transitions.
 * Use sparingly - only at domain boundaries.
 */
#define lb_smp_mb() smp_mb()

#endif /* _KERNEL_SCHED_FAIR_OPT_H */

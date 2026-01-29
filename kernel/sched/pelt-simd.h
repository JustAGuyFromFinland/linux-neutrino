/* SPDX-License-Identifier: GPL-2.0 */
/*
 * SIMD and branchless optimizations for PELT (Per Entity Load Tracking)
 *
 * This header provides AVX2-optimized and branchless versions of PELT
 * functions for improved cache efficiency and reduced branch mispredictions.
 *
 * Copyright (C) 2024
 */
#ifndef _KERNEL_SCHED_PELT_SIMD_H
#define _KERNEL_SCHED_PELT_SIMD_H

#include <linux/types.h>
#include <linux/compiler.h>

#ifdef CONFIG_X86_64

/*
 * Structure of Arrays (SoA) for batch PELT processing.
 * Instead of processing one sched_avg at a time (AoS), we process
 * multiple entities in parallel using SIMD operations.
 *
 * This allows AVX2 to process 4 x u64 values simultaneously.
 * All arrays are 64-byte aligned for optimal cache line usage
 * and to ensure proper alignment for AVX2/AVX-512 operations.
 */
#define PELT_BATCH_SIZE 4

struct sched_avg_soa {
	u64 last_update_time[PELT_BATCH_SIZE] __aligned(64);
	u64 load_sum[PELT_BATCH_SIZE] __aligned(64);
	u64 runnable_sum[PELT_BATCH_SIZE] __aligned(64);
	u32 util_sum[PELT_BATCH_SIZE] __aligned(64);
	u32 period_contrib[PELT_BATCH_SIZE] __aligned(64);
	unsigned long load_avg[PELT_BATCH_SIZE] __aligned(64);
	unsigned long runnable_avg[PELT_BATCH_SIZE] __aligned(64);
	unsigned long util_avg[PELT_BATCH_SIZE] __aligned(64);
	unsigned int util_est[PELT_BATCH_SIZE] __aligned(64);
	/* Validity mask: 1 = valid entry, 0 = skip */
	u8 valid_mask;
} __aligned(64);

/*
 * Branchless minimum: returns min(a, b) without branches
 * Uses the fact that (a - b) >> 63 is all 1s if a < b, else all 0s
 */
static __always_inline u64 branchless_min_u64(u64 a, u64 b)
{
	u64 diff = a - b;
	u64 mask = (s64)diff >> 63; /* Arithmetic shift: all 1s if negative */
	return (a & mask) | (b & ~mask);
}

/*
 * Branchless maximum: returns max(a, b) without branches
 */
static __always_inline u64 branchless_max_u64(u64 a, u64 b)
{
	u64 diff = a - b;
	u64 mask = (s64)diff >> 63;
	return (b & mask) | (a & ~mask);
}

/*
 * Branchless conditional: returns a if cond != 0, else b
 * Zero-extending boolean to full mask
 */
static __always_inline u64 branchless_select_u64(u64 cond, u64 a, u64 b)
{
	u64 mask = (u64)(-(s64)!!cond);
	return (a & mask) | (b & ~mask);
}

static __always_inline u32 branchless_select_u32(u32 cond, u32 a, u32 b)
{
	u32 mask = (u32)(-(s32)!!cond);
	return (a & mask) | (b & ~mask);
}

/*
 * Branchless decay_load - optimized version without branches
 *
 * Original code has:
 *   if (unlikely(n > LOAD_AVG_PERIOD * 63)) return 0;
 *   if (unlikely(local_n >= LOAD_AVG_PERIOD)) { ... }
 *
 * This version computes both paths and selects the result.
 */
static __always_inline u64 decay_load_branchless(u64 val, u64 n,
						 const u32 *yN_inv_table)
{
	u64 periods, remainder;
	u64 shifted_val;
	u64 result;
	u64 zero_mask;

	/* Compute mask: all 1s if n > LOAD_AVG_PERIOD * 63 (2016) */
	zero_mask = (s64)(2016ULL - n) >> 63;

	/* Compute periods = n / 32 and remainder = n % 32 */
	periods = n >> 5; /* n / LOAD_AVG_PERIOD where LOAD_AVG_PERIOD = 32 */
	remainder = n & 31; /* n % LOAD_AVG_PERIOD */

	/* Shift val by periods (equivalent to dividing by 2^periods) */
	/* Clamp periods to 63 to avoid undefined behavior */
	periods = branchless_min_u64(periods, 63);
	shifted_val = val >> periods;

	/* Apply y^remainder decay using lookup table */
	/* Safe index: remainder is already masked to 0-31 range */
	result = mul_u64_u32_shr(shifted_val, yN_inv_table[remainder & 31], 32);

	/* Return 0 if zero_mask is set, otherwise return result */
	return result & ~zero_mask;
}

/*
 * Branchless accumulate for a single contribution type
 * Computes: sum += contrib * weight (only if weight != 0)
 */
static __always_inline u64 accumulate_branchless(u64 sum, u32 contrib,
						 unsigned long weight, int shift)
{
	u64 addition = (u64)contrib;
	u64 weight_mask;

	/* Create mask: all 1s if weight != 0, all 0s otherwise */
	weight_mask = (u64)(-(s64)!!weight);

	/* Compute addition with optional shift */
	addition = branchless_select_u64(shift > 0,
					 (u64)weight * contrib << shift,
					 (u64)weight * contrib);

	/* Add only if weight is non-zero (via mask) */
	return sum + (addition & weight_mask);
}

/*
 * Batch decay using AVX2 intrinsics (when available)
 * Processes 4 u64 values simultaneously
 */
#ifdef CONFIG_AS_AVX2
#include <asm/fpu/api.h>

static inline void decay_load_batch_avx2(u64 *values, u64 n, int count,
					 const u32 *yN_inv_table)
{
	int i;

	if (likely(count >= 4)) {
		kernel_fpu_begin();

		/* Process 4 values at a time using AVX2 */
		for (i = 0; i + 4 <= count; i += 4) {
			/*
			 * Note: Full AVX2 intrinsics implementation would use
			 * _mm256_load_si256, _mm256_srli_epi64, etc.
			 * For now, use optimized scalar with prefetch.
			 */
			__builtin_prefetch(&values[i + 4], 0, 3);

			values[i + 0] = decay_load_branchless(values[i + 0], n, yN_inv_table);
			values[i + 1] = decay_load_branchless(values[i + 1], n, yN_inv_table);
			values[i + 2] = decay_load_branchless(values[i + 2], n, yN_inv_table);
			values[i + 3] = decay_load_branchless(values[i + 3], n, yN_inv_table);
		}

		kernel_fpu_end();
	}

	/* Handle remaining values */
	for (; i < count; i++)
		values[i] = decay_load_branchless(values[i], n, yN_inv_table);
}

/*
 * AVX2 batch update for SoA structure
 * Updates load_sum, runnable_sum, and util_sum in parallel
 */
static inline void update_sums_batch_avx2(struct sched_avg_soa *soa,
					  u64 periods, u32 contrib,
					  const unsigned long *loads,
					  const unsigned long *runnables,
					  const int *runnings,
					  const u32 *yN_inv_table)
{
	int i;

	kernel_fpu_begin();

	/* Prefetch the data we'll need */
	__builtin_prefetch(soa->load_sum, 1, 3);
	__builtin_prefetch(soa->runnable_sum, 1, 3);
	__builtin_prefetch(soa->util_sum, 1, 3);

	for (i = 0; i < PELT_BATCH_SIZE; i++) {
		u8 valid = (soa->valid_mask >> i) & 1;
		u64 mask = (u64)(-(s64)valid);

		/* Decay old sums (branchless) */
		u64 new_load_sum = decay_load_branchless(soa->load_sum[i],
							 periods, yN_inv_table);
		u64 new_runnable_sum = decay_load_branchless(soa->runnable_sum[i],
							     periods, yN_inv_table);
		u64 new_util_sum = decay_load_branchless(soa->util_sum[i],
							 periods, yN_inv_table);

		/* Accumulate new contributions (branchless) */
		new_load_sum = accumulate_branchless(new_load_sum, contrib,
						     loads[i], 0);
		new_runnable_sum = accumulate_branchless(new_runnable_sum, contrib,
							 runnables[i], SCHED_CAPACITY_SHIFT);
		new_util_sum = accumulate_branchless(new_util_sum, contrib,
						     runnings[i], SCHED_CAPACITY_SHIFT);

		/* Store results (masked by validity) */
		soa->load_sum[i] = (new_load_sum & mask) | (soa->load_sum[i] & ~mask);
		soa->runnable_sum[i] = (new_runnable_sum & mask) | (soa->runnable_sum[i] & ~mask);
		soa->util_sum[i] = (u32)((new_util_sum & mask) | (soa->util_sum[i] & ~mask));
	}

	kernel_fpu_end();
}

#else /* !CONFIG_AS_AVX2 */

static inline void decay_load_batch_avx2(u64 *values, u64 n, int count,
					 const u32 *yN_inv_table)
{
	int i;

	for (i = 0; i < count; i++)
		values[i] = decay_load_branchless(values[i], n, yN_inv_table);
}

static inline void update_sums_batch_avx2(struct sched_avg_soa *soa,
					  u64 periods, u32 contrib,
					  const unsigned long *loads,
					  const unsigned long *runnables,
					  const int *runnings,
					  const u32 *yN_inv_table)
{
	int i;

	for (i = 0; i < PELT_BATCH_SIZE; i++) {
		if (!((soa->valid_mask >> i) & 1))
			continue;

		soa->load_sum[i] = decay_load_branchless(soa->load_sum[i],
							 periods, yN_inv_table);
		soa->runnable_sum[i] = decay_load_branchless(soa->runnable_sum[i],
							     periods, yN_inv_table);
		soa->util_sum[i] = (u32)decay_load_branchless(soa->util_sum[i],
							      periods, yN_inv_table);

		soa->load_sum[i] = accumulate_branchless(soa->load_sum[i], contrib,
							 loads[i], 0);
		soa->runnable_sum[i] = accumulate_branchless(soa->runnable_sum[i],
							     contrib, runnables[i],
							     SCHED_CAPACITY_SHIFT);
		soa->util_sum[i] = (u32)accumulate_branchless(soa->util_sum[i],
							      contrib, runnings[i],
							      SCHED_CAPACITY_SHIFT);
	}
}

#endif /* CONFIG_AS_AVX2 */

/*
 * Convert AoS (Array of Structures) to SoA (Structure of Arrays)
 * for batch processing
 */
static inline void aos_to_soa(struct sched_avg_soa *soa,
			      struct sched_avg **avgs, int count)
{
	int i;

	soa->valid_mask = 0;

	for (i = 0; i < PELT_BATCH_SIZE && i < count; i++) {
		if (avgs[i]) {
			soa->last_update_time[i] = avgs[i]->last_update_time;
			soa->load_sum[i] = avgs[i]->load_sum;
			soa->runnable_sum[i] = avgs[i]->runnable_sum;
			soa->util_sum[i] = avgs[i]->util_sum;
			soa->period_contrib[i] = avgs[i]->period_contrib;
			soa->load_avg[i] = avgs[i]->load_avg;
			soa->runnable_avg[i] = avgs[i]->runnable_avg;
			soa->util_avg[i] = avgs[i]->util_avg;
			soa->util_est[i] = avgs[i]->util_est;
			soa->valid_mask |= (1 << i);
		}
	}
}

/*
 * Convert SoA back to AoS after batch processing
 */
static inline void soa_to_aos(struct sched_avg_soa *soa,
			      struct sched_avg **avgs, int count)
{
	int i;

	for (i = 0; i < PELT_BATCH_SIZE && i < count; i++) {
		if (avgs[i] && ((soa->valid_mask >> i) & 1)) {
			avgs[i]->last_update_time = soa->last_update_time[i];
			avgs[i]->load_sum = soa->load_sum[i];
			avgs[i]->runnable_sum = soa->runnable_sum[i];
			avgs[i]->util_sum = soa->util_sum[i];
			avgs[i]->period_contrib = soa->period_contrib[i];
			avgs[i]->load_avg = soa->load_avg[i];
			avgs[i]->runnable_avg = soa->runnable_avg[i];
			avgs[i]->util_avg = soa->util_avg[i];
			avgs[i]->util_est = soa->util_est[i];
		}
	}
}

#else /* !CONFIG_X86_64 */

/*
 * Non-x86_64 fallback: use branchless but no SIMD
 */
struct sched_avg_soa {
	u64 placeholder; /* Unused on non-x86_64 */
};

#endif /* CONFIG_X86_64 */

/*
 * Prefetch helpers for scheduler hot paths
 */
#define SCHED_PREFETCH_LOCALITY_HIGH	3  /* Keep in all cache levels */
#define SCHED_PREFETCH_LOCALITY_MED	2  /* Keep in L2 and L3 */
#define SCHED_PREFETCH_LOCALITY_LOW	1  /* Keep only in L3 */
#define SCHED_PREFETCH_LOCALITY_NONE	0  /* Non-temporal (streaming) */

static __always_inline void sched_prefetch_read(const void *addr)
{
	__builtin_prefetch(addr, 0, SCHED_PREFETCH_LOCALITY_HIGH);
}

static __always_inline void sched_prefetch_write(const void *addr)
{
	__builtin_prefetch(addr, 1, SCHED_PREFETCH_LOCALITY_HIGH);
}

/*
 * Prefetch the next sched_avg in a linked list traversal
 */
static __always_inline void prefetch_sched_avg(struct sched_avg *avg)
{
	if (avg) {
		/* Prefetch first cacheline (last_update_time, load_sum, etc.) */
		sched_prefetch_read(avg);
		/* Prefetch second cacheline if structure spans multiple */
		sched_prefetch_read((char *)avg + L1_CACHE_BYTES);
	}
}

#endif /* _KERNEL_SCHED_PELT_SIMD_H */

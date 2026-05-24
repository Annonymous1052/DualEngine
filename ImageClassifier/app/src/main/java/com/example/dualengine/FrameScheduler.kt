package com.example.dualengine

import android.os.SystemClock
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Latency-aware frame scheduler (paper §IV-E).
 * TD(t) = max(O_pre / f_CPU, O_inf / f_GPU)
 * TL(t) = mean + std of offloaded frame latencies in the previous time slot.
 * Routes frames to offload with probability TD / (TD + TL).
 */
class FrameScheduler {

    @Volatile
    var modelIndex: Int = 0
        private set

    @Volatile
    var tdMs: Double = DEFAULT_TD_MS
        private set

    @Volatile
    var tlMs: Double = DEFAULT_TL_MS
        private set

    private val latenciesCurrentSlot = mutableListOf<Double>()
    private val latenciesPreviousSlot = mutableListOf<Double>()
    private val lock = Any()

    fun updateControl(modelIdx: Int, cpuFreqHz: Long, gpuFreqHz: Long) {
        val m = modelIdx.coerceIn(0, OPRE_CYCLES.size - 1)
        modelIndex = m
        val fCpu = max(cpuFreqHz, MIN_FREQ_HZ)
        val fGpu = max(gpuFreqHz, MIN_FREQ_HZ)
        tdMs = max(
            OPRE_CYCLES[m] * MS_PER_SEC / fCpu,
            OINF_CYCLES[m] * MS_PER_SEC / fGpu
        ).coerceAtLeast(MIN_TD_MS)
    }

    /** Call at the start of each measurement time slot. */
    fun beginTimeslot() {
        synchronized(lock) {
            latenciesPreviousSlot.clear()
            latenciesPreviousSlot.addAll(latenciesCurrentSlot)
            latenciesCurrentSlot.clear()
            tlMs = computeTlLocked()
        }
    }

    private val pendingSendTimesMs = ArrayDeque<Long>()

    /** Call when an offloaded frame is written to the socket (non-blocking). */
    fun onOffloadSend() {
        synchronized(lock) {
            pendingSendTimesMs.addLast(SystemClock.uptimeMillis())
        }
    }

    /**
     * Call when the server ACKs a completed offloaded inference (FIFO).
     * @return true if the frame counted toward offload FPS (RTT <= TL).
     */
    fun onOffloadAck(): Boolean {
        synchronized(lock) {
            val sendTime = pendingSendTimesMs.removeFirstOrNull() ?: return false
            val rttMs = SystemClock.uptimeMillis() - sendTime
            latenciesCurrentSlot.add(rttMs.toDouble())
            return if (rttMs <= tlMs) {
                completedOffloads++
                true
            } else {
                droppedFrames++
                false
            }
        }
    }

    /** Fraction of frames sent to the edge server: TD / (TD + TL). */
    fun offloadFraction(): Double {
        val denom = tdMs + tlMs
        if (denom <= 0.0) return 0.5
        return (tdMs / denom).coerceIn(0.0, 1.0)
    }

    fun shouldRouteToOffload(): Boolean {
        return Random.nextDouble() < offloadFraction()
    }

    fun isWithinLatencyThreshold(latencyMs: Long): Boolean {
        return latencyMs.toDouble() <= tlMs
    }

    private var droppedFrames = 0
    private var completedOffloads = 0

    fun droppedCount(): Int = synchronized(lock) { droppedFrames }
    fun completedOffloadCount(): Int = synchronized(lock) { completedOffloads }

    fun resetSessionCounters() {
        synchronized(lock) {
            droppedFrames = 0
            completedOffloads = 0
            latenciesCurrentSlot.clear()
            latenciesPreviousSlot.clear()
            pendingSendTimesMs.clear()
        }
        tlMs = DEFAULT_TL_MS
        tdMs = DEFAULT_TD_MS
    }

    private fun computeTlLocked(): Double {
        if (latenciesPreviousSlot.isEmpty()) {
            return DEFAULT_TL_MS
        }
        val mean = latenciesPreviousSlot.average()
        if (latenciesPreviousSlot.size == 1) {
            return mean.coerceAtLeast(MIN_TL_MS)
        }
        val variance = latenciesPreviousSlot.map { (it - mean).pow(2) }.average()
        return (mean + sqrt(variance)).coerceAtLeast(MIN_TL_MS)
    }

    companion object {
        private const val MS_PER_SEC = 1000.0
        private const val MIN_FREQ_HZ = 1L
        private const val MIN_TD_MS = 1.0
        private const val MIN_TL_MS = 1.0

        const val DEFAULT_TL_MS = 150.0
        const val DEFAULT_TD_MS = 50.0

        /** Profiled cycle counts for yolov8s / m / x (preprocess, inference). */
        private val OPRE_CYCLES = longArrayOf(8_000_000L, 20_000_000L, 35_000_000L)
        private val OINF_CYCLES = longArrayOf(80_000_000L, 250_000_000L, 600_000_000L)
    }
}

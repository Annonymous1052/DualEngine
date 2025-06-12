package com.example.dualengine

import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.annotation.NonNull
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.model.Model
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import kotlin.math.max
import kotlin.math.min

class DQNAgent(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val trainingSamples: MutableList<TrainingSample> = mutableListOf()
    private var executor: ExecutorService? = null
    private val handler = Handler(Looper.getMainLooper())

    fun close() {
        interpreter = null
    }

    private fun setupModelPersonalization() {
        val options = Interpreter.Options()
//        options.numThreads = numThreads
        val modelFile = FileUtil.loadMappedFile(context, "model.tflite")
        interpreter = Interpreter(modelFile, options)
    }

    fun addSample(state: FloatArray, action: FloatArray, reward: FloatArray, nextState: FloatArray) {
        if (interpreter == null) {
            setupModelPersonalization()
        }
        if (state.size != NUM_STATE) {
            Log.d("ERROR", "wrong state size")
        }
        if (action.size != NUM_ACTION) {
            Log.d("ERROR", "wrong action size")
        }
        if (reward.size != NUM_ACTION) {
            Log.d("ERROR", "wrong reward size")
        }
        if (nextState.size != NUM_STATE) {
            Log.d("ERROR", "wrong nextState size")
        }
        trainingSamples.add(
            TrainingSample(state, action, reward, nextState)
        )
    }

    fun startTraining(): Float {
        if (interpreter == null) {
            setupModelPersonalization()
        }

        // Create new thread for training process.
        val trainBatchSize = getTrainBatchSize()

        // Will modify later to determine training status from outside.
//        if (trainingSamples.size < trainBatchSize) {
//            throw RuntimeException(
//                String.format(
//                    "Too few samples to start training: need %d, got %d",
//                    trainBatchSize, trainingSamples.size
//                )
//            )
//        }

        val avgLoss: Float
        var totalLoss = 0f
        var numBatchesProcessed = 0

        // Shuffle training samples
        Log.d("ONGOING", "TrainingSamples shuffle")
        trainingSamples.shuffle()

        trainingBatches(trainBatchSize).forEach {
                trainingSamples ->
            val trainingBatchStates: MutableList<FloatArray> = ArrayList(NUM_STATE)
            val trainingBatchActions: MutableList<FloatArray> = ArrayList(NUM_ACTION)
            val trainingBatchRewards: MutableList<FloatArray> = ArrayList(NUM_ACTION)
            val trainingBatchNextStates: MutableList<FloatArray> = ArrayList(NUM_STATE)
            // input training lists.
            Log.d("ONGOING", "TrainingSamples make batch")
            trainingSamples.forEach{ trainingSample ->
                trainingBatchStates.add(trainingSample.state)
                trainingBatchActions.add(trainingSample.action)
                trainingBatchRewards.add(trainingSample.reward)
                trainingBatchNextStates.add(trainingSample.nextState)
            }
//            Log.d("ONGOING", "TrainingSamples batch size: ${trainingBatchStates.size}")
//            Log.d("ONGOING", "TrainingSamples state size: ${trainingBatchStates[0].size}")
//            Log.d("ONGOING", "TrainingSamples training")
            val loss = trainArray(
                trainingBatchStates,
                trainingBatchActions,
                trainingBatchRewards,
                trainingBatchNextStates
            )
            totalLoss += loss
            numBatchesProcessed++
        }

        // Calculate the average loss after training all batches.
        avgLoss = totalLoss / numBatchesProcessed
        return avgLoss
//        handler.post {
//            classifierListener?.onLossResults(avgLoss)
//        }
    }

    fun floatArrayToFloatBuffer(floatArray: FloatArray): FloatBuffer {
        // Calculate FloatBuffer size
        val bufferSize = floatArray.size * 4 // Each float takes 4 bytes

        // Create ByteBuffer
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
            .order(ByteOrder.nativeOrder())

        // Convert ByteBuffer to FloatBuffer
        val floatBuffer = byteBuffer.asFloatBuffer()

        // Copy FloatArray to FloatBuffer
        floatBuffer.put(floatArray)
        floatBuffer.position(0) // Set buffer position to the beginning

        return floatBuffer
    }

    fun predict(state: TensorBuffer): MutableList<Float> {
        if (interpreter == null) {
            setupModelPersonalization()
        }
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[INFERENCE_INPUT_KEY] = state.buffer

//        val output = TensorBuffer.createFixedSize(intArrayOf(1, 256), DataType.FLOAT32)
        val outputs: MutableMap<String, Any> = HashMap()
        val output = TensorBuffer.createFixedSize(
            intArrayOf(1, 256),
            DataType.FLOAT32
        )
        outputs[INFERENCE_OUTPUT_KEY] = output.buffer

//        var inferenceTime = SystemClock.uptimeMillis()
        interpreter?.runSignature(inputs, outputs, INFERENCE_KEY)
//        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
//        Log.d("TEST", "inference time: $inferenceTime")
        return output.floatArray.toMutableList() //.getValue(INFERENCE_OUTPUT_KEY)
//        classifierListener?.onResults(inferenceTime)
    }

    fun predictTarget(state: TensorBuffer): Any {
        if (interpreter == null) {
            setupModelPersonalization()
        }
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[INFERENCE_TARGET_INPUT_KEY] = state.buffer

//        val output = TensorBuffer.createFixedSize(intArrayOf(1, 256), DataType.FLOAT32)
        val outputs: MutableMap<String, Any> = HashMap()
        val output = TensorBuffer.createFixedSize(
            intArrayOf(1, 256),
            DataType.FLOAT32
        )
        outputs[INFERENCE_TARGET_OUTPUT_KEY] = output.buffer

//        var inferenceTime = SystemClock.uptimeMillis()
        interpreter?.runSignature(inputs, outputs, INFERENCE_TARGET_KEY)
//        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
//        Log.d("TEST", "inference time: $inferenceTime")
        return output.floatArray.toList() //.getValue(INFERENCE_OUTPUT_KEY)
//        classifierListener?.onResults(inferenceTime)
    }

    fun train(state: TensorBuffer, action: TensorBuffer, reward: TensorBuffer, nextState: TensorBuffer): Any {
        if (interpreter == null) {
            setupModelPersonalization()
        }
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[TRAINING_INPUT_STATE_KEY] = state.buffer
        inputs[TRAINING_INPUT_ACTION_KEY] = action.buffer
        inputs[TRAINING_INPUT_REWARD_KEY] = reward.buffer
        inputs[TRAINING_INPUT_NEXT_STATE_KEY] = nextState.buffer

        val outputs: MutableMap<String, Any> = HashMap()
        val output = TensorBuffer.createFixedSize(
            intArrayOf(1),
            DataType.FLOAT32
        )
        outputs[TRAINING_OUTPUT_KEY] = output.buffer

        interpreter?.runSignature(inputs, outputs, TRAINING_KEY)

        return output
    }

    fun trainArray(state: MutableList<FloatArray>, action: MutableList<FloatArray>, reward: MutableList<FloatArray>, nextState: MutableList<FloatArray>): Float {
        if (interpreter == null) {
            setupModelPersonalization()
        }
//        Log.d("ONGOING", "state size: ${state.size}")
//        Log.d("ONGOING", "state unit size: ${state[0].size}")
//
//        Log.d("ONGOING", "action size: ${action.size}")
//        Log.d("ONGOING", "action unit size: ${action[0].size}")
//
//        Log.d("ONGOING", "reward size: ${reward.size}")
//        Log.d("ONGOING", "reward unit size: ${reward[0].size}")
//
//        Log.d("ONGOING", "nextState size: ${nextState.size}")
//        Log.d("ONGOING", "nextState unit size: ${nextState[0].size}")
//        for (i in 0..<EXPECTED_BATCH_SIZE) {
//            val inputs: MutableMap<String, Any> = HashMap()
//            inputs[TRAINING_INPUT_STATE_KEY] = state[i]
//            inputs[TRAINING_INPUT_ACTION_KEY] = action[i]
//            inputs[TRAINING_INPUT_REWARD_KEY] = reward[i]
//            inputs[TRAINING_INPUT_NEXT_STATE_KEY] = nextState[i]
//
//            val outputs: MutableMap<String, Any> = HashMap()
//            val output = TensorBuffer.createFixedSize(
//                intArrayOf(1),
//                DataType.FLOAT32
//            )
//            outputs[TRAINING_OUTPUT_KEY] = output.buffer
//
//            interpreter?.runSignature(inputs, outputs, TRAINING_KEY)
//            val loss = output.getFloatValue(0)
////            Log.d("TEST", "From trainArray, loss: $loss")
//        }
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[TRAINING_INPUT_STATE_KEY] = state.toTypedArray()
        inputs[TRAINING_INPUT_ACTION_KEY] = action.toTypedArray()
        inputs[TRAINING_INPUT_REWARD_KEY] = reward.toTypedArray()
        inputs[TRAINING_INPUT_NEXT_STATE_KEY] = nextState.toTypedArray()

        val outputs: MutableMap<String, Any> = HashMap()
        val output = TensorBuffer.createFixedSize(
            intArrayOf(1),
            DataType.FLOAT32
        )
        outputs[TRAINING_OUTPUT_KEY] = output.buffer
//            val loss = FloatBuffer.allocate(1)
//            outputs[TRAINING_OUTPUT_KEY] = loss
        Log.d("ONGOING", "TrainingSamples trainArray")
        interpreter?.runSignature(inputs, outputs, TRAINING_KEY)
//        Log.d("ONGOING", "TrainingSamples trainArray finished")
        Log.d("ONGOING", "TrainingSamples trainArray result: ${output.floatArray.joinToString(",")}")
//            val loss2 = output.getFloatValue(0)
        return output.getFloatValue(0)
    }

    fun update_target(): Any {
        if (interpreter == null) {
            setupModelPersonalization()
        }
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[UPDATE_INPUT_KEY] = "tmp"

        val outputs: MutableMap<String, Any> = HashMap()
        val output = TensorBuffer.createFixedSize(
            intArrayOf(1, 1),
            DataType.FLOAT32
        )
        outputs[UPDATE_OUTPUT_KEY] = output.buffer

//        var inferenceTime = SystemClock.uptimeMillis()
        interpreter?.runSignature(inputs, outputs, UPDATE_KEY)
//        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
//        Log.d("TEST", "inference time: $inferenceTime")
        return output.floatArray.toList()
//        classifierListener?.onResults(inferenceTime)
    }

    fun update_lr(lr: Float): Any {
        if (interpreter == null) {
            setupModelPersonalization()
        }
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[UPDATE_LR_INPUT_KEY] = lr

        val outputs: MutableMap<String, Any> = HashMap()
        val output = TensorBuffer.createFixedSize(
            intArrayOf(1, 1),
            DataType.FLOAT32
        )
        outputs[UPDATE_LR_OUTPUT_KEY] = output.buffer

        interpreter?.runSignature(inputs, outputs, UPDATE_LR_KEY)
        return output.floatArray.toList()
    }

    fun returnWeight(): List<List<Float>> {
        if (interpreter == null) {
            setupModelPersonalization()
        }
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[RETURN_WEIGHT_INPUT_KEY] = "tmp"
        val checkpointPath = "/data/local/tmp/model_saved.ckpt"
//        val output = TensorBuffer.createFixedSize(intArrayOf(1, 256), DataType.FLOAT32)
        val outputs: MutableMap<String, Any> = HashMap()
        val outputW = TensorBuffer.createFixedSize(
            intArrayOf(NUM_STATE, NUM_ACTION),
            DataType.FLOAT32
        )
        val outputB = TensorBuffer.createFixedSize(
            intArrayOf(81,),
            DataType.FLOAT32
        )
        outputs[RETURN_WEIGHT_OUTPUT_W_KEY] = outputW.buffer
        outputs[RETURN_WEIGHT_OUTPUT_B_KEY] = outputB.buffer
        interpreter?.runSignature(inputs, outputs, RETURN_WEIGHT_KEY)
        return listOf(outputW.floatArray.toList(), outputB.floatArray.toList())
    }

    fun save(w: List<Float>, b: List<Float>): Any {
        if (interpreter == null) {
            setupModelPersonalization()
        }
        val weightPath = "model_saved_weight.txt"
        val biasPath = "model_saved_bias.txt"
        // Will save received weights and bias to file. How to do this?

        val weightFile = File(context.filesDir, weightPath)
        weightFile.parentFile?.mkdirs()
        weightFile.printWriter().use {out ->
            w.forEach { out.println(it) }
        }

        val biasFile = File(context.filesDir, biasPath)
        biasFile.parentFile?.mkdirs()
        biasFile.printWriter().use {out ->
            b.forEach { out.println(it) }
        }
        return 1
    }

    fun restore(weights: List<Any>): Any {
        if (interpreter == null) {
            setupModelPersonalization()
        }
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[RESTORE_INPUT_W_KEY] = weights[0]
        inputs[RESTORE_INPUT_B_KEY] = weights[1]

        val outputs: MutableMap<String, Any> = HashMap()
        val output = TensorBuffer.createFixedSize(
            intArrayOf(1, 1),
            DataType.FLOAT32
        )
        outputs[RESTORE_OUTPUT_KEY] = output.buffer
        interpreter?.runSignature(inputs, outputs, RESTORE_KEY)
        return 1
    }

    fun load(): List<Any> {
        if (interpreter == null) {
            setupModelPersonalization()
        }
        val weightPath = "model_saved_weight.txt"
        val biasPath = "model_saved_bias.txt"

        val weightFile = File(context.filesDir, weightPath)
        var w = weightFile.readLines().map { it.toFloat() }
        var wArray = reshapeTo2DArray(w, NUM_STATE, NUM_ACTION)

        val biasFile = File(context.filesDir, biasPath)
        var b = biasFile.readLines().map { it.toFloat() }
        var bArray = b.toFloatArray()

        return listOf(wArray, bArray)
    }

    fun reshapeTo2DArray(floatList: List<Float>, rows: Int, cols: Int): Array<FloatArray> {
        require(floatList.size == rows * cols) {
            "The size of the list does not match the provided shape"
        }

        val reshapedArray = Array(rows) { FloatArray(cols) }
        for (i in floatList.indices) {
            reshapedArray[i / cols][i % cols] = floatList[i]
        }
        return reshapedArray
    }

    fun listToTensorBuffer(floatList: List<Float>, shape: IntArray): TensorBuffer {
        require(floatList.size == shape.reduce { acc, i -> acc * i }) {
            "The size of the list does not match the provided shape"
        }
        val tensorBuffer = TensorBuffer.createFixedSize(shape, DataType.FLOAT32)
        val floatArray = floatList.toFloatArray()
        tensorBuffer.loadArray(floatArray)

        return tensorBuffer
    }

    interface ClassifierListener {
        fun onError(error: String)
        fun onResults(inferenceTime: Long)
        fun onLossResults(lossNumber: Float)
    }

    fun getTrainBatchSize(): Int {
        return min(
            max( /* at least one sample needed */1, trainingSamples.size),
            EXPECTED_BATCH_SIZE
        )
    }

    fun getExpectedBatchSize(): Int {
        return EXPECTED_BATCH_SIZE
    }

    private fun trainingBatches(trainBatchSize: Int): Iterator<MutableList<TrainingSample>> {
        return object : Iterator<MutableList<TrainingSample>> {
            private var nextIndex = 0

            override fun hasNext(): Boolean {
                return nextIndex < trainingSamples.size
            }

            override fun next(): MutableList<TrainingSample> {
                val fromIndex = nextIndex
                val toIndex: Int = nextIndex + trainBatchSize
                nextIndex = toIndex
                return if (toIndex >= trainingSamples.size) {
                    // To keep batch size consistent, last batch may include some elements from the
                    // next-to-last batch.
                    trainingSamples.subList(
                        trainingSamples.size - trainBatchSize,
                        trainingSamples.size
                    )
                } else {
                    trainingSamples.subList(fromIndex, toIndex)
                }
            }
        }
    }

    companion object {
        private const val EXPECTED_BATCH_SIZE = 20
        private const val NUM_STATE = 15
        private const val NUM_ACTION = 81

        private const val TRAINING_INPUT_STATE_KEY = "state"
        private const val TRAINING_INPUT_ACTION_KEY = "action"
        private const val TRAINING_INPUT_REWARD_KEY = "reward"
        private const val TRAINING_INPUT_NEXT_STATE_KEY = "next_state"
        private const val TRAINING_OUTPUT_KEY = "output_0"
        private const val TRAINING_KEY = "train"

        private const val INFERENCE_INPUT_KEY = "state"
        private const val INFERENCE_OUTPUT_KEY = "output"
        private const val INFERENCE_KEY = "infer"

        private const val INFERENCE_TARGET_INPUT_KEY = "state"
        private const val INFERENCE_TARGET_OUTPUT_KEY = "output"
        private const val INFERENCE_TARGET_KEY = "infer_target"

        private const val UPDATE_INPUT_KEY = "tmp"
        private const val UPDATE_OUTPUT_KEY = "output_0"
        private const val UPDATE_KEY = "update_target"

        private const val UPDATE_LR_INPUT_KEY = "tmp"
        private const val UPDATE_LR_OUTPUT_KEY = "output_0"
        private const val UPDATE_LR_KEY = "update_lr"


        private const val RESTORE_INPUT_B_KEY = "bias"
        private const val RESTORE_INPUT_W_KEY = "weight"
        private const val RESTORE_OUTPUT_KEY = "output_0"
        private const val RESTORE_KEY = "restore"

        private const val RETURN_WEIGHT_INPUT_KEY = "checkpoint_path"
        private const val RETURN_WEIGHT_OUTPUT_W_KEY = "output_0"
        private const val RETURN_WEIGHT_OUTPUT_B_KEY = "output_1"
        private const val RETURN_WEIGHT_KEY = "return_weights"

    }
    data class TrainingSample(val state: FloatArray, val action: FloatArray, val reward: FloatArray, val nextState: FloatArray)
}


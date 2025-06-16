package com.example.dualengine

import android.content.Context
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.annotation.NonNull
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.model.Model
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Queue

/* Classifier class implementation using TensorFlow Lite Support Library's model.Model class */
class ClassifierWithModel(context: Context, private val MODEL_NAME: String) {
    /* Constant declarations */
    companion object{
        // 2-1. Model loading: Add tflite model to assets directory
        // 2-2. Model loading: Declare model filename as constant
        // 5-1. Inference result interpretation: Add txt file containing classification class labels to assets directory
        // 5-2. Inference result interpretation: Declare label filename as constant

        //        private const val MODEL_NAME = "yolov8l-cls_float32.tflite"
        private const val LABEL_FILE = "imagenet_labels.txt"
    }
    /* Property declarations */
    var context: Context = context

    // ===========================================================================================
    // 2-4. Model loading: interpreter property declaration
    // lateinit var interpreter: Interpreter
    // When using Model class, no need to create Interpreter directly
    var model: Model
    // ============================================================================================

    // 3-1. Input image preprocessing: Property declaration to store model's input image
    lateinit var inputImage: TensorImage
    // 3-2. Input image preprocessing: Model's input shape property declarations
    var modelInputChannel: Int = 0
    var modelInputWidth: Int = 0
    var modelInputHeight: Int = 0
    // 4-1. Inference: Property declaration to store model's inferred output values
    lateinit var outputBuffer: TensorBuffer
    // 5-3. Inference result interpretation: Property declaration to store label list
    private lateinit var labels: List<String>


//    val resources: Resources = context.resources
//    val bitmap_image = BitmapFactory.decodeResource(resources, R.drawable.cat)

    init{
        // ========================================================================================
        // 2-3. Model loading: Load tflite file
        // val model: ByteBuffer? = FileUtil.loadMappedFile(context, MODEL_NAME)
        // model?.order(ByteOrder.nativeOrder())?:throw IOException()
        // interpreter = Interpreter(model)
        // Model class performs everything from tflite file loading to inference
        //model = Model.createModel(context, MODEL_NAME)
        // Model performance improvement
        //model = createMultiThreadModel(2) // CPU multi-thread model
        var loadstart = SystemClock.uptimeMillis()
        model = createGPUModel()          // GPU delegate model
        // model = createNNAPIModel()        // NNAPI delegate model
        // ========================================================================================

        // 3-4. Input image preprocessing: Method call
        initModelShape()
        var loadLatency = SystemClock.uptimeMillis()-loadstart
        Log.d("TAG", "Model load latency: $loadLatency")


        // 5-4. Inference result interpretation: Load label file

        loadstart = SystemClock.uptimeMillis()
        labels = FileUtil.loadLabels(context, LABEL_FILE)
        loadLatency = SystemClock.uptimeMillis()-loadstart
        Log.d("TAG", "Label load latency: $loadLatency")
    }

    // 3-3. Input image preprocessing: Method definition
    // Store model's input shape and data type in properties
    private fun initModelShape(){
        // ========================================================================================
        // val inputTensor = interpreter.getInputTensor(0)
        val inputTensor = model.getInputTensor(0)
        // ========================================================================================
        val shape = inputTensor.shape()
        modelInputChannel = shape[0]
        modelInputWidth = shape[1]
        modelInputHeight = shape[2]
        // Create TensorImage to store model's input values
        inputImage = TensorImage(inputTensor.dataType())

        // 4-2. Inference: Create TensorBuffer to store model's output values
        // ========================================================================================
        // val outputTensor = interpreter.getOutputTensor(0)
        val outputTensor = model.getOutputTensor(0)
        // ========================================================================================
        outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType())

    }

    // 3-4. Input image preprocessing: TensorImage bitmap image input and image preprocessing logic definition
    // Receives Bitmap image as input, preprocesses it and returns it in TensorImage format
    fun loadImage(bitmap: Bitmap?): TensorImage{
        // Store image data in TensorImage
        // 7-2. Addition - Data format conversion: Convert if bitmap's data format is not ARGB_8888
        if (bitmap != null) {
            if(bitmap.config != Bitmap.Config.ARGB_8888)
                inputImage.load(convertBitmap2ARGB8888(bitmap))
            else
                inputImage.load(bitmap)
        }
        // inputImage?.load(bitmap)

        // Define preprocessing ImageProcessor
        val imageProcessor =
            ImageProcessor.Builder()                            // Create Builder
                .add(ResizeOp(modelInputWidth,modelInputHeight, // Image size conversion
                    ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(NormalizeOp(0.0f, 255.0f))    // Image normalization
                .build()                                       // Create ImageProcessor
        // Preprocess image and return in TensorImage format
        return imageProcessor.process(inputImage)
    }

    // 7-1. Addition - Data format conversion: Convert if bitmap variable in loadImage method is not ARGB_8888
    private fun convertBitmap2ARGB8888(bitmap: Bitmap): Bitmap{
        return bitmap.copy(Bitmap.Config.ARGB_8888, true)
    }

    // 4-3. Inference: Define inference method
    fun classify(image: TensorImage): Pair<String, Float>{
        // ========================================================================================
        // interpreter.run(inputImage.buffer, outputBuffer.buffer.rewind())
        // Model class parameters require Object array and Object Map respectively
        val inputs = arrayOf<Any>(image.buffer as Any)
        val outputs = mutableMapOf<Int, Any>()
        outputs[0] = outputBuffer.buffer.rewind() as Any
        model.run(inputs, outputs as @receiver:NonNull Map<Int, Any>)
        // ========================================================================================

        // 5-5. Inference result interpretation: Map model output values to labels and return
        val output = "example" to 0.0f // Map<String, Float>  // pose estimation; PJ
        return output
//        val output = TensorLabel(labels, outputBuffer).getMapWithFloatValue() // Map<String, Float>  // object detection; PJ
//        return argmax(output)
    }

    // 5-6. Inference result interpretation: Method definition to find and return the class name and probability pair with highest probability from Map
    private fun argmax(map: Map<String, Float>): Pair<String, Float>{
        var maxKey = ""
        var maxVal = -1.0f

        for(entry in map.entries){
            var f = entry.value
            if(f > maxVal){
                maxKey = entry.key
                maxVal = f
            }
        }

        return Pair(maxKey, maxVal)
    }

    // Inference performance improvement: CPU multi-thread
    private fun createMultiThreadModel(nThreads: Int): Model{
        try {
            val optionsBuilder = Model.Options.Builder()
            optionsBuilder.setNumThreads(nThreads)
            return Model.createModel(context, MODEL_NAME, optionsBuilder.build())
        }catch(ioe: IOException){
            throw ioe
        }
    }

    // Inference performance improvement: GPU delegation
    private fun createGPUModel(): Model {
        try {
            val optionsBuilder = Model.Options.Builder()
            val compatList = CompatibilityList()

            if (compatList.isDelegateSupportedOnThisDevice) {
                Log.d("TAG", "GPU Use")
                optionsBuilder.setDevice(Model.Device.GPU)
            }
            else
                Log.d("TAG", "GPU not support")

            return Model.createModel(context, MODEL_NAME, optionsBuilder.build())
        }catch(ioe: IOException){
            throw ioe
        }
    }

    // Inference performance improvement: NNAPI delegation
    private fun createNNAPIModel(): Model{
        try{
            val optionsBuilder = Model.Options.Builder()

            if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P)
                optionsBuilder.setDevice(Model.Device.NNAPI)

            return Model.createModel(context, MODEL_NAME, optionsBuilder.build())
        }catch(ioe: IOException){
            throw ioe
        }
    }

    // 6. Resource cleanup: Define resource cleanup method
    fun finish(){
        // ========================================================================================
        // if(interpreter != null)
        //     interpreter.close()
        if(model != null)
            model.close()
        // ========================================================================================
    }

}
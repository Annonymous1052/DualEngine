package com.example.dualengine

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder


/* Classifier class implementation using TensorFlow Lite Support Library */
class ClassifierWithSuppport(context: Context) {
    /* Constant declarations */
    companion object{
        // 2-1. Model loading: Add tflite model to assets directory
        // 2-2. Model loading: Declare model filename as constant
        private const val MODEL_NAME = "mobilenet_float16tflite"
        // 5-1. Inference result interpretation: Add txt file containing classification class labels to assets directory
        // 5-2. Inference result interpretation: Declare label filename as constant
        private const val LABEL_FILE = "imagenet_labels.txt"
    }
    /* Property declarations */
    var context: Context = context
    // 2-4. Model loading: interpreter property declaration
    var interpreter: Interpreter?
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


    init{
        // 2-3. Model loading: Load tflite file
        val model: ByteBuffer? = FileUtil.loadMappedFile(context, MODEL_NAME)
        model?.order(ByteOrder.nativeOrder())?:throw IOException()
        interpreter = Interpreter(model)
        // Model performance improvement
        // interpreter = createMultiThreadInterpreter(2) // CPU multi-thread
        // interpreter = createGPUInterpreter()          // GPU delegation
        // interpreter = createNNAPIInterpreter()        // NNAPI delegation model
        // 3-4. Input image preprocessing: Method call
        initModelShape()
        // 5-4. Inference result interpretation: Load label file
        labels = FileUtil.loadLabels(context, LABEL_FILE)
    }

    // 3-3. Input image preprocessing: Method definition
    // Store model's input shape and data type in properties
    private fun initModelShape(){
        val inputTensor = interpreter!!.getInputTensor(0)
        val shape = inputTensor.shape()
        modelInputChannel = shape[0]
        modelInputWidth = shape[1]
        modelInputHeight = shape[2]
        // Create TensorImage to store model's input values
        inputImage = TensorImage(inputTensor.dataType())

        // 4-2. Inference: Create TensorBuffer to store model's output values
        val outputTensor = interpreter!!.getOutputTensor(0)
        outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType())

    }

    // 3-5. Input image preprocessing: TensorImage bitmap image input and image preprocessing logic definition
    // Receives Bitmap image as input, preprocesses it and returns it in TensorImage format
    private fun loadImage(bitmap: Bitmap): TensorImage{
        // Store image data in TensorImage
        inputImage.load(bitmap)
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

    // 4-3. Inference: Define inference method
    fun classify(image: Bitmap): Pair<String, Float>{
        inputImage = loadImage(image)
        if (interpreter == null){
            return Pair("Error", 0f)
        }
        interpreter!!.run(inputImage.buffer, outputBuffer.buffer.rewind())

        // 5-5. Inference result interpretation: Map model output values to labels and return
        val output = TensorLabel(labels, outputBuffer).getMapWithFloatValue() // Map<String, Float>

        return argmax(output)

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

    // Model performance improvement: CPU multi-thread
    private fun createMultiThreadInterpreter(nThreads: Int): Interpreter{
        try{
            val options = Interpreter.Options()
            options.setNumThreads(nThreads)
            val model = FileUtil.loadMappedFile(context, MODEL_NAME)
            model.order(ByteOrder.nativeOrder())
            return Interpreter(model, options)
        }catch(ioe: IOException){
            throw ioe
        }
    }

    // Model performance improvement: GPU delegation
    private fun createGPUInterpreter(): Interpreter{
        try{
            val options = Interpreter.Options()
            val compatList = CompatibilityList()

            if(compatList.isDelegateSupportedOnThisDevice){
                val delegateOptions = compatList.bestOptionsForThisDevice
                val gpuDelegate = GpuDelegate(delegateOptions)
                //val gpuDelegate = GpuDelegateFactory(delegateOptions)
                options.addDelegate(gpuDelegate)
            }

            val model = FileUtil.loadMappedFile(context, MODEL_NAME)
            model.order(ByteOrder.nativeOrder())
            return Interpreter(model, options)
        }catch(ioe: IOException){
            throw ioe
        }
    }

    // Model performance improvement: NNAPI delegation
    private fun createNNAPIInterpreter(): Interpreter{
        try{
            val options = Interpreter.Options()

            if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.P){
                val nnApiDelegate = NnApiDelegate()
                options.addDelegate(nnApiDelegate)
            }

            val model = FileUtil.loadMappedFile(context, MODEL_NAME)
            model.order(ByteOrder.nativeOrder())
            return Interpreter(model, options)
        }catch(ioe: IOException){
            throw ioe
        }
    }


    // 6. Resource cleanup: Define resource cleanup method
    fun finish(){
        interpreter?.close()
    }

}
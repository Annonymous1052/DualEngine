// main
package com.example.dualengine

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.Bundle
import android.os.Process
import android.os.SystemClock
import android.telephony.TelephonyManager
import android.util.Base64
import android.util.Log
import android.view.WindowManager
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.FileProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ReceiveChannel
import kotlinx.coroutines.channels.SendChannel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import com.example.dualengine.databinding.ActivityPhotoBinding
import okio.utf8Size
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.io.OutputStream
import java.io.OutputStreamWriter
import java.io.PrintWriter
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.util.Collections.max
import kotlin.math.tanh
import kotlin.random.Random


class PhotoActivity : BaseActivity() {
    /* Property declarations */
    val binding by lazy{ActivityPhotoBinding.inflate(layoutInflater)}

    var photoUri: Uri? = null
    // Model property declarations
    lateinit var classifier_s: ClassifierWithModel
    lateinit var classifier_m: ClassifierWithModel
    //    lateinit var classifier_l: ClassifierWithModel
    lateinit var classifier_x: ClassifierWithModel
    val ones = listOf(
        1,
        4,
        7,
        10,
        13,
        16,
        19,
        22,
        25,
        28,
        31,
        34,
        37,
        40,
        43,
        46,
        49,
        52,
        55,
        58,
        61,
        64,
        67,
        70,
        73,
        76,
        79
    )  // action with model m
    val twos = listOf(
        2,
        5,
        8,
        11,
        14,
        17,
        20,
        23,
        26,
        29,
        32,
        35,
        38,
        41,
        44,
        47,
        50,
        53,
        56,
        59,
        62,
        65,
        68,
        71,
        74,
        77,
        80
    )  // action with model x
    //    val gpu_list = listOf("130000", "221000", "312000", "403000", "494000", "585000", "676000", "767000", "858000")  // 0~8
//    val cpu0_list = listOf("400000", "533000", "650000", "754000", "858000", "962000", "1066000", "1170000", "1274000", "1378000",
//        "1482000", "1586000", "1690000", "1794000", "1898000", "2002000", "2106000", "2210000")  // 0~17
//    val cpu4_list = listOf("533000", "624000", "728000", "832000", "936000", "1040000", "1144000", "1248000", "1352000", "1456000",
//        "1560000", "1664000", "1768000", "1872000", "1976000", "2080000", "2184000", "2288000", "2392000",
//        "2496000", "2600000", "2704000", "2808000")  // 0~22
    val gpu_list = listOf("221000", "494000", "858000")  //("403000", "676000", "858000")
    val cpu0_list = listOf("754000", "1482000", "2210000")  //("1066000", "1690000", "2210000")
    val cpu4_list = listOf("832000", "1768000", "2808000")  //("1248000", "2080000", "2808000")
    val off_list = listOf(4, 8, 12)  // number of offload images
    val acc_list = listOf(72.3, 76.4, 78.4)  //
    lateinit var job: Job
//    var model_set = mutableListOf<ClassifierWithModel>()

    val actionChannel = Channel<List<Float>>(1) // action from DQN
    val imageInfChannel = Channel<TensorImage>(10) // images to run local inference
    val imageOffChannel = Channel<String>(10) // images to offload
    val measureChannel = Channel<List<Float>>(1)  // background measurement data
    val FPSOffChannel = Channel<String>(1)  // FPS data
    //    val FPSOffChannel = Channel<String>(100)  // FPS data noR
    val FPSInfChannel = Channel<String>(1)  // FPS data
    val modelChannel = Channel<List<Float>>(1)  // selected model for inference
    val rChannel = Channel<Int>(1)  // selected offloading rate
    val rChannelChecker = Channel<Int>(1)  // for safe-action of r
    val finishChannel = Channel<Int>(1)  // if not empty, finish experiment
    val startChannel = Channel<Int>(1)   // if not empty, start experiment. - For synchronizing inference and offloading.

    // parameters for DQN
    val timeslotSize = 2000  // in ms units.
    val trainedModel = 0  // whether to use trained model
    val dqnAgent = DQNAgent(this)
    val lambda = 100f  // weight for Heat reward  [80, 100, 120 으로 실험한 이후 다음 값 보기.. ]
    val kappa = 8f  // weight for Memory reward
    val mu = 50f  // weight for FPS reward
    val nu = 5f  // weight for accuracy reward
    val nuB = 350f  // bias for accuracy reward
    val h_th = 62f  // thermal threshold
    val m_th = 500f  // memory threshold
    val targetFPS = 20f  // FPS threshold
    var epsilon = 1f  // for epsilon-greedy
    var epsilonDecay = 0.95f
    val epsilonReset = 30  // reset epsilon every n timeslot
    var epsilonCnt = 0 // for counting epsilon
    var lr = 0.2f
    var lrReset = 0.2f
    var lrMax = 1f
    var lrCnt = 0
    var lrResetCnt = 30 // reset lr every n timeslot
    var lrdecay = 0.95f
    var lrmin = 0.0001f
    val num_sample = 1 // number of sample to add upon low reward

    val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission())
    { isGranted: Boolean ->
        if (isGranted) {
            //
        } else {
            // Explain to the user that the feature is unavailable because the
            // feature requires a permission that the user has denied. At the
            // same time, respect the user's decision. Don't link to system
            // settings in an effort to convince the user to change their
            // decision.
        }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(binding.root)
        val contextForCoroutine = this  // For passing Context to coroutines.
        binding.imageView.setImageResource(R.drawable.cat)
        binding.editTextTime.isEnabled = true
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_PHONE_STATE) == PackageManager.PERMISSION_GRANTED) {
            // Code to execute when permission is already granted
        } else {
            // Display permission request dialog when permission is not granted
            requestPermissionLauncher.launch(
                Manifest.permission.READ_PHONE_STATE)
        }
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            // Code to execute when permission is already granted
        } else {
            // Display permission request dialog when permission is not granted
            requestPermissionLauncher.launch(
                Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            // Code to execute when permission is already granted
        } else {
            // Display permission request dialog when permission is not granted
            requestPermissionLauncher.launch(
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }

        // Button listeners
        binding.btnStart.setOnClickListener {
            Log.d("TAG", "start button call")
            var expTime = binding.editTextTime.text.toString().toInt()
            val addr = "192.168.0.2" //"192.168.0.2" "14.4.61.130"
            Log.d("TAG", "run $expTime second")

            // Communication socket for offloading
            val thread = SocketThread(addr, expTime)
            Log.d("so","thread start")
            thread.start()  // for offloading

            // Coroutines for on-device and image preprocessing, measurement, memory load
            runBlocking() {
                job = launch { imageCoroutine(imageInfChannel, imageOffChannel, finishChannel) }
                launch { inferCoroutine(imageInfChannel, FPSInfChannel, modelChannel, startChannel, rChannelChecker, finishChannel, contextForCoroutine) } // for local
                launch { measureCoroutine(FPSInfChannel, FPSOffChannel, modelChannel, rChannel, startChannel, rChannelChecker, finishChannel, contextForCoroutine) }
                if(!binding.memStart1.text.isEmpty()){
                    launch {
                        delay(binding.memStart1.text.toString().toInt()*1000L)
                        Log.d("TAG", "mem1 started")
                        executeMemCommand(listOf("su", "-c", "./stress-ng", "--vm", "1", "--vm-bytes", binding.memSize1.text.toString()+"M", "-t", binding.memDuration1.text.toString()+"s --page-in"))
                        //val test = executeCommand("su -c awk '{print \$1}' /proc/meminfo")
                        //Log.d("TAG", test)
                    }
                }
                if(!binding.memStart2.text.isEmpty()){
                    launch {
                        delay(binding.memStart2.text.toString().toInt()*1000L)
                        Log.d("TAG", "mem2 started")
                        executeMemCommand(listOf("su", "-c", "./stress-ng", "--vm", "1", "--vm-bytes", binding.memSize2.text.toString()+"M", "-t", binding.memDuration2.text.toString()+"s --page-in"))
                    }
                }
            }
        }

        binding.sendButton.setOnClickListener {
            // Share experiment data
            binding.textViewtest.text = "send button clicked!"
            val fileName = "data_file.txt"
            val file = File(applicationContext.filesDir, fileName)

            // Convert file to content URI
            val contentUri: Uri = FileProvider.getUriForFile(applicationContext, "com.example.dualengine.fileprovider", file)
            val intent = Intent(Intent.ACTION_SEND)
            intent.type = "text/plain"
            intent.putExtra(Intent.EXTRA_STREAM, contentUri)
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            startActivity(Intent.createChooser(intent, "Share!"))
        }

        binding.sendWeight.setOnClickListener {
            // Share experiment data
            binding.textViewtest.text = "send weight clicked!"
            val fileName = "model_saved_weight.txt"
            val file = File(applicationContext.filesDir, fileName)

            // Convert file to content URI
            val contentUri: Uri = FileProvider.getUriForFile(applicationContext, "com.example.dualengine.fileprovider", file)
            val intent = Intent(Intent.ACTION_SEND)
            intent.type = "text/plain"
            intent.putExtra(Intent.EXTRA_STREAM, contentUri)
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            startActivity(Intent.createChooser(intent, "Share!"))
        }

        binding.sendBias.setOnClickListener {
            // Share experiment data
            binding.textViewtest.text = "send bias clicked!"
            val fileName = "model_saved_bias.txt"
            val file = File(applicationContext.filesDir, fileName)

            // Convert file to content URI
            val contentUri: Uri = FileProvider.getUriForFile(applicationContext, "com.example.dualengine.fileprovider", file)
            val intent = Intent(Intent.ACTION_SEND)
            intent.type = "text/plain"
            intent.putExtra(Intent.EXTRA_STREAM, contentUri)
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            startActivity(Intent.createChooser(intent, "Share!"))
        }

        // Load models
        try{
            classifier_s = ClassifierWithModel(this, "yolov8s-cls_float32.tflite")
            classifier_m = ClassifierWithModel(this, "yolov8m-cls_float32.tflite")
            classifier_x = ClassifierWithModel(this, "yolov8x-cls_float32.tflite")

        }catch(ioe: IOException){
            ioe.printStackTrace()
        }
    }

    inner class SocketThread(var host: String,var duration: Int) : Thread() {
        override fun run() {
            runBlocking() {
                launch { commCoroutine(host, imageOffChannel, FPSOffChannel, rChannel, startChannel, finishChannel) }
            }
        }
    }

    override fun permissionGranted(requestCode: Int) {}
    override fun permissionDenied(requestCode: Int) {}
    fun readFileToBase64(filePath: String): String {
        val file = File(filePath)

        try {
            FileInputStream(file).use { fileInputStream ->
                val byteArray = ByteArray(file.length().toInt())
                fileInputStream.read(byteArray)

                return Base64.encodeToString(byteArray, Base64.DEFAULT)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }

        return ""
    }
    //override fun onDestroy() {
    //    super.onDestroy()
    //    classifier.finish()
    //}

    fun executeCommand(command: String): String {
        val process = Runtime.getRuntime().exec(command)
        val reader = BufferedReader(InputStreamReader(process.inputStream))
        val output = StringBuilder()

        var line: String?
        while (reader.readLine().also { line = it } != null) {
            output.append(line).append('\n')
        }

        reader.close()

        // Wait for process termination.
        val exitCode = process.waitFor()

        // Return result if command executed successfully.
        return if (exitCode == 0) {
            output.toString()
        } else {
            // Return error code if command failed.
            "Error executing command: $command\nExit code: $exitCode"
        }
    }
    fun executeMemCommand(command: List<String>) {
        try {
            // Set command to execute and working directory
            val workingDirectory = "/data/local/tmp"

            // Create ProcessBuilder and set working directory
            val processBuilder = ProcessBuilder(command)
            processBuilder.directory(File(workingDirectory))

            // Execute command
            val process = processBuilder.start()

            // Output results
//            val reader = BufferedReader(InputStreamReader(process.inputStream))
//            val output = StringBuilder()
//
//            var line: String?
//            while (reader.readLine().also { line = it } != null) {
//                output.append(line).append('\n')
//            }
//
//            reader.close()
//            // Wait for process termination
//            val exitCode = process.waitFor()
//            Log.d("TAG", output.toString())
//            Log.d("TAG","Exit Code: $exitCode")
//            val reader2 = BufferedReader(InputStreamReader(process.errorStream))
//            val output2 = StringBuilder()
//
//            var line2: String?
//            while (reader2.readLine().also { line2 = it } != null) {
//                output2.append(line2).append('\n')
//            }
//
//            reader2.close()
//            Log.d("TAG", output2.toString())

        } catch (e: IOException) {
            e.printStackTrace()
        } catch (e: InterruptedException) {
            e.printStackTrace()
        }
    }
    fun setFreq(freqs: List<Int>){
//        var gpu_list = listOf("130000", "221000", "312000", "403000", "494000", "585000", "676000", "767000", "858000")  // 0~8
//        var cpu0_list = listOf("400000", "533000", "650000", "754000", "858000", "962000", "1066000", "1170000", "1274000", "1378000",
//            "1482000", "1586000", "1690000", "1794000", "1898000", "2002000", "2106000", "2210000")  // 0~17
//        var cpu4_list = listOf("533000", "624000", "728000", "832000", "936000", "1040000", "1144000", "1248000", "1352000", "1456000",
//            "1560000", "1664000", "1768000", "1872000", "1976000", "2080000", "2184000", "2288000", "2392000",
//            "2496000", "2600000", "2704000", "2808000")  // 0~22

        var CPU_f_0 = cpu0_list[freqs[0]]
        var CPU_f_4 = cpu4_list[freqs[1]]
        var GPU_f = gpu_list[freqs[2]]

        var startTime = SystemClock.uptimeMillis()
        Log.d("TAG", "changing freqs as $CPU_f_0, $CPU_f_4, $GPU_f")
        executeCommand("su -c echo $CPU_f_0 > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq; " +
                "su -c echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor; " +
                "su -c echo $CPU_f_4 > /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq;" +
                "su -c echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor; " +
                "su -c echo $CPU_f_4 > /sys/devices/system/cpu/cpufreq/policy7/scaling_max_freq;" +
                "su -c echo performance > /sys/devices/system/cpu/cpufreq/policy7/scaling_governor; " +
                "su -c echo $GPU_f > /sys/kernel/gpu/gpu_max_clock;" +
                "su -c echo $GPU_f > /sys/kernel/gpu/gpu_min_clock;")

        var endTime = (SystemClock.uptimeMillis() - startTime)/1000
        Log.d("TAG", "changing freqs finished! with latency $endTime")
    }
    fun getDataName(): String {
        var testData = executeCommand("su -c cat /sys/devices/virtual/thermal/thermal_zone*/type")
        val pid = Process.myPid()
        testData = testData.replace("\n", ",")
        testData = "timestamp,ModelL,ModelX,Model_num,FPSInf,FPSOff,FPStotal,acc,cpu0,cpu0,cpu4,cpu4,gpu,gpu,r,loss,reward_heat,reward_memory,reward_FPS,reward_acc,reward,lr,"+testData.substring(0, testData.length-1) + ",GPU_min,GPU_max,CPU0_max,CPU0_freq,CPU4_max,CPU4_freq,CPU7_max,CPU7_freq," +executeCommand("su -c awk '{print $1}' /proc/meminfo")+executeCommand("su -c awk '{print \$1}' /proc/$pid/status | grep -E 'VmRSS|VmSwap'")+"Down,Up,Type,"
        testData = testData.replace("\n", ",")+ '\n'
        return testData
    }
    fun getMeasure(): String {
        var info : String
        val pid = Process.myPid()
        var tempString = executeCommand("su -c awk '{print \$1/1000}' /sys/devices/virtual/thermal/thermal_zone*/temp; " +
                "awk '{print \$1/1000}' /sys/kernel/gpu/gpu_min_clock; awk '{print \$1/1000}' /sys/kernel/gpu/gpu_max_clock; " +
                "awk '{print \$1/1000}' /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq; awk '{print \$1/1000}' /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; " +
                "awk '{print \$1/1000}' /sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq; awk '{print \$1/1000}' /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq; " +
                "awk '{print \$1/1000}' /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq; awk '{print \$1/1000}' /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq; " +
                "awk '{print \$2/1024}' /proc/meminfo; " +
                "grep -E 'VmRSS|VmSwap' /proc/$pid/status | awk '{print $2/1024}'")
        tempString = tempString.replace("\n", ",")

        Log.d("ONGOING", "else")
        // Connectivity Manager
        val cm = applicationContext.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        // Network Capabilities of Active Network
        val nc = cm.getNetworkCapabilities(cm.activeNetwork)
        val nw = cm.activeNetwork
        val actNw = cm.getNetworkCapabilities(nw)
        when {
            actNw!!.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> info = "Wifi"
            actNw.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> info = "Ethernet"
            actNw.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> {
                val tm =
                    applicationContext.getSystemService(Context.TELEPHONY_SERVICE) as TelephonyManager
                when (tm.dataNetworkType) {
                    TelephonyManager.NETWORK_TYPE_GPRS,
                    TelephonyManager.NETWORK_TYPE_EDGE,
                    TelephonyManager.NETWORK_TYPE_CDMA,
                    TelephonyManager.NETWORK_TYPE_1xRTT,
                    TelephonyManager.NETWORK_TYPE_IDEN,
                    TelephonyManager.NETWORK_TYPE_GSM -> info = "2G"

                    TelephonyManager.NETWORK_TYPE_UMTS,
                    TelephonyManager.NETWORK_TYPE_EVDO_0,
                    TelephonyManager.NETWORK_TYPE_EVDO_A,
                    TelephonyManager.NETWORK_TYPE_HSDPA,
                    TelephonyManager.NETWORK_TYPE_HSUPA,
                    TelephonyManager.NETWORK_TYPE_HSPA,
                    TelephonyManager.NETWORK_TYPE_EVDO_B,
                    TelephonyManager.NETWORK_TYPE_EHRPD,
                    TelephonyManager.NETWORK_TYPE_HSPAP,
                    TelephonyManager.NETWORK_TYPE_TD_SCDMA -> info = "3G"

                    TelephonyManager.NETWORK_TYPE_LTE, 19,
                    TelephonyManager.NETWORK_TYPE_IWLAN -> info = "4G"

                    TelephonyManager.NETWORK_TYPE_NR -> info = "5G"
                    else -> info = "None"
                }
            }
            else -> info = "None"
        }

        var downSpeed = (nc!!.linkDownstreamBandwidthKbps) / 1000  // DownSpeed in MBPS
        var upSpeed = (nc.linkUpstreamBandwidthKbps) / 1000  // UpSpeed  in MBPS

        return tempString.substring(0, tempString.length - 1) + "," + downSpeed.toString() + "," + upSpeed.toString() + ",$info" + '\n'
    }
    fun getReward(h: Float, m: Float, model: Float, FPSInf: Float, FPSOff: Float): List<Float> {
        var FPS = FPSInf + FPSOff
        var heatReward = 0f
        var memoryReward = 0f
        var FPSReward = 0f
        var accReward = 0f
        var accAverage = 0f

        // Heat
        if (h >= h_th) {
            heatReward = -lambda
        } else {
            heatReward = tanh(h_th - h) * lambda
        }

        // Memory
        if (m <= m_th) {
            memoryReward = -kappa
        } else {
            memoryReward = (m / m_th) * kappa
        }

        // FPS
        if (FPS <= targetFPS) {
            FPSReward = -mu * (targetFPS / FPS)
        } else {
            FPSReward = mu
        }

        // Accuracy
        accAverage = ((FPSInf * acc_list[model.toInt()] + FPSOff * acc_list[2]) / FPS).toFloat()
        accReward = nu * accAverage - nuB

        // Return total reward
        return listOf(heatReward + memoryReward + FPSReward + accReward, heatReward, memoryReward, FPSReward, accReward, accAverage)
    }
    fun getAction(states: FloatArray): Int? {
        // predict with model.tflite
        Log.d("ONGOING", "states: " + states.joinToString(","))
        Log.d("ONGOING", "tensor making")
        val tensorBuffer = TensorBuffer.createDynamic(DataType.FLOAT32)
        tensorBuffer.loadArray(states, intArrayOf(states.size))
        Log.d("ONGOING", "predict")
        var result = dqnAgent.predict(tensorBuffer)
        Log.d("TEST", "Predicted result: $result")
        // Remove actions for models that are not loaded.
        var l1 = states[10]
        var l2 = states[11]
        var action = 0 // initialize
        if (l1 == 0f) {
            for (i in ones) {
                result[i] = Float.NEGATIVE_INFINITY
            }
        }
        if (l2 == 0f) {
            for (i in twos) {
                result[i] = Float.NEGATIVE_INFINITY
            }
        }

        Log.d("ONGOING", "successfully ran DQN!")
        // epsilon greedy
        if (Random.nextFloat() < epsilon) {
            // Random action
            Log.d("Action", "Random action selected")
            action = Random.nextInt(0,81)
            while (result[action] == Float.NEGATIVE_INFINITY) {
                action = Random.nextInt(0,81)
            }
        } else {
            // Normal action with biggest value
            action = result.subList(0,81).argmax()!!
        }
        epsilon = epsilonDecay*epsilon
        if (epsilon < 0.01f) {
            epsilon = 0f
        }

        // Select and return the highest value among remaining actions (= currently selectable actions)

        Log.d("Action", "ActionList: ${result.subList(0,81)}")
        return action
    }
    fun toBase3List(number: Int?): List<Int> {
        require(number in 0..80) { "Number must be between 0 and 80" }

        if (number == null) return listOf(0, 0, 0, 0)

        var num = number
        val base3List = mutableListOf<Int>()

        while (num > 0) {
            base3List.add(num % 3)
            num /= 3
        }

        while (base3List.size < 4) {
            base3List.add(0)
        }

        return base3List.reversed()
    }
    fun fromBase3List(base3List: List<Int>): Int {
        require(base3List.size == 4) { "List must contain exactly 4 elements" }
        require(base3List.all { it in 0..2 }) { "Each element must be between 0 and 2" }

        var number = 0
        for (i in base3List.indices) {
            number = number * 3 + base3List[i]
        }
        return number
    }
    fun updateLR(lr: Float): Any {
        dqnAgent.update_lr(lr)
        return 0
    }
    suspend fun CoroutineScope.imageCoroutine(channelInf: SendChannel<TensorImage>, channelSend: SendChannel<String>, channelFinish: ReceiveChannel<Int>){
        // Role: Preprocess and supply images for on-device AI and MEC offloading
        // Interactions:
        // Supply images for on-device AI processing through channelInf
        // Supply images for MEC offloading processing through channelSend
        // Items that need modification: X

        runBlocking {
            launch {// send to server
                var inputImage = readFileToBase64("/data/local/tmp/cat5.jpg")
                while (channelFinish.isEmpty) {channelSend.send(inputImage)}
            }
            launch {// send to local
                while (channelFinish.isEmpty) {
                    var inputImage = readFileToBase64("/data/local/tmp/cat5.jpg")
                    val imageBytes = Base64.decode(inputImage, 0)
                    val image = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                    var inputImage2 = classifier_s.loadImage(image)
                    channelInf.send(inputImage2)
                }
            }
        }
        //Cleanup after experiment ends
        var inputImage = readFileToBase64("/data/local/tmp/cat5.jpg")
        val imageBytes = Base64.decode(inputImage, 0)
        val image = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        var inputImage2 = classifier_s.loadImage(image)  // Especially this part.

        delay(1000)
        channelSend.send(inputImage)
        channelInf.send(inputImage2)

        Log.d("Finished", "imageCoroutine finished")
    }
    suspend fun CoroutineScope.inferCoroutine(channelImage: ReceiveChannel<TensorImage>, channelFPS: SendChannel<String>, channelModel: Channel<List<Float>>, channelStart: ReceiveChannel<Int>, channelRChecker: ReceiveChannel<Int>, channelFinish: ReceiveChannel<Int>, context: Context){
        // Role: Perform on-device AI
        // Interactions:
        // Receive images for inference from imageCoroutine through channelImage.
        // Receive model number to use for inference from measureCoroutine through channelModel.
        // Send average FPS, used model set size, and used model to measureCoroutine through channelFPS every second.
        // Items that need modification:
        // Need to modify the model usage part. Simplified to 3 types: s, m, x.
        var FPS = 0.0
        var model_num = 2
        var l1 = 1
        var l2 = 1
        var addTime = 0L  // Time taken from FPS transmission to action reception.
        while(channelStart.isEmpty){
            delay(10)
        }
        var startTime = SystemClock.uptimeMillis()
        var prev = startTime

        while(channelFinish.isEmpty){
            var queueImage = channelImage.receive()

            // Create model queue and use appropriate model when available.
            if (!channelModel.isEmpty){
                var recv = channelModel.receive()
                l1 = recv[0].toInt()
                l2 = recv[1].toInt()
                model_num = recv[2].toInt()

                if (model_num==3) {
                    // For continuous inference during model loading.
                    while(channelModel.isEmpty) {
                        classifier_s.classify(queueImage)
                        FPS += 1
                    }
                    var recv = channelModel.receive()
                    l1 = recv[0].toInt()
                    l2 = recv[1].toInt()
                    model_num = recv[2].toInt()
                }
                addTime = SystemClock.uptimeMillis() - prev
                prev = SystemClock.uptimeMillis() // When receiving new model, execute additionally for timeslot size from there.
            }

            val output: Pair<String, Float>
            when (model_num) {
                0 -> output = classifier_s.classify(queueImage)
                1 -> output = classifier_m.classify(queueImage)
                2 -> output = classifier_x.classify(queueImage)
            }
            FPS += 1
            // Create FPS channel to share.
            if (SystemClock.uptimeMillis() - prev > timeslotSize && !channelRChecker.isEmpty) {
                channelRChecker.receive()
                var timeStamp = ((SystemClock.uptimeMillis()-startTime)/1000.0).toString()
                var startChannel = SystemClock.uptimeMillis()
                var FPSsend = FPS/(SystemClock.uptimeMillis() - prev + addTime)*1000

                channelFPS.send(timeStamp + "," + l1 + "," + l2 + "," + model_num + "," + FPSsend.toString())
                Log.d("Channel", "channelFPS send latency at inferCoroutine: ${SystemClock.uptimeMillis() - startChannel}, current model $model_num")
                Log.d("TIME", "Time stamp: $timeStamp")
                prev = SystemClock.uptimeMillis()
                Log.d("FPS", "FPSInf: $FPSsend")
                FPS = 0.0
                addTime = 0L
            }
        }
        while(channelImage.isEmpty == false) {
            var queueImage = channelImage.receive() // Cleanup after experiment ends.
        }
        Log.d("Finished", "inferCoroutine finished")
    }
    suspend fun CoroutineScope.commCoroutine(host: String, channelImage: ReceiveChannel<String>, channelFPS: SendChannel<String>, channelR: ReceiveChannel<Int>, channelStart: ReceiveChannel<Int>, channelFinish: ReceiveChannel<Int>){
        // Role: MEC offloading. Send images to server.
        // Interactions: From imageCoroutine through channelImage
        //
        try {
            val port = 5000 //Port number same as server side
            Log.d("testLog","host: $host")
//                var testResult = executeCommand("adb ")
            val socket = Socket(host, port)
            val out: OutputStream = socket.getOutputStream()
            val writer = PrintWriter(OutputStreamWriter(out, StandardCharsets.UTF_8))
            val input: InputStream = socket.getInputStream()
            val reader = BufferedReader(InputStreamReader(input, StandardCharsets.UTF_8))
            Log.d("testLog","server connected!")
            // Send image
            var b64Image = channelImage.receive()
            var imageSize = b64Image.utf8Size()

            Log.d("testLog", "imagesize : $imageSize")
            writer.println(imageSize)
            writer.flush()
            var response = reader.readLine() // Get response "ok"
            Log.d("testLog", "response : $response")

            var r = 15
            var FPS = 0

            while(channelStart.isEmpty){
                delay(10)
            }
            var startTime = SystemClock.uptimeMillis()
            var prev = startTime

            while(channelFinish.isEmpty){
                // from source
                if (FPS == r) {
                    delay(10)
                } else {
                    b64Image = channelImage.receive()

                    writer.println(b64Image)
                    writer.flush()
                    FPS += 1
                }

                if (SystemClock.uptimeMillis() - prev > timeslotSize) {
//                    var timeStamp = ((SystemClock.uptimeMillis()-startTime)/1000.0).toString()
                    var FPSsend = FPS/((SystemClock.uptimeMillis()-prev)/1000)
                    var startChannel = SystemClock.uptimeMillis()
                    channelFPS.send(FPSsend.toString())
                    var endChannel = SystemClock.uptimeMillis() - startChannel
                    r=channelR.receive()*timeslotSize/1000  // noR
                    Log.d("FPS", "FPSOff: $FPSsend")
                    Log.d("Channel", "channelFPS send and receive latency at commCoroutine: $endChannel")
                    prev = SystemClock.uptimeMillis()
                    FPS = 0
                }
            }
            writer.println("\n\n")
            writer.flush()

            Log.d("Finished", "finished. close socket")
            socket.close() // Release socket
            while(channelImage.isEmpty == false) {
                var queueImage = channelImage.receive() // Cleanup after experiment ends.
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        Log.d("Finished", "commCoroutine finished")
    }
    suspend fun CoroutineScope.measureCoroutine(channelInfFPS: ReceiveChannel<String>, channelOffFPS: ReceiveChannel<String>, channelModel: SendChannel<List<Float>>, channelR: SendChannel<Int>, channelStart: SendChannel<Int>, channelRChecker: SendChannel<Int>, channelFinish: SendChannel<Int>, context: Context) {
        // Role: Identify current device state and determine appropriate decision variables, save results
        // Interactions:
        // Receive FPS inferred by on-device AI through channelInfFPS.
        // Receive FPS inferred through MEC offloading through channelOffFPS.
        // Send model number to use for on-device AI through channelModel.
        // Send number of images to transmit through MEC offloading through channelR.
        // Items that need modification:
        // Reflect DQN-based decision variable selection algorithm
        // Add training.
        // Modify model memory loading part


        var result: MutableList<String> = mutableListOf()
        var startTime = SystemClock.uptimeMillis()
        var duration = binding.editTextTime.text.toString().toInt()  // Experiment time
        var tempString : String

//        var alg = "noM"  // HM: proposed method, noR: no offloading rate control, noM: no model set control, noH: no thermal control; mem control only

        var cpu0 = 2f  // current freq of cpu0
        var cpu4 = 2f  // current freq of cpu4
        var gpu = 2f  // current freq of gpu


        var h_CPU0 = 0.0f  // current temperature of CPU
        var h_CPU4 = 0.0f  // current temperature of CPU
        var h_CPU7 = 0.0f  // current temperature of CPU
        var h_GPU = 0.0f  // current temperature of GPU
        var h_5G = 0.0f  // current temperature of 5G

        var m_available = 0.0f  // current available memory

        var r = 2f  // target offloaded frames
        var r_sent = 15.0f  // actual offloaded frames

        // model information
        var l1_prev = 1f
        var l2_prev = 1f
        var l1 = 1f // Whether L model is loaded
        var l2 = 1f // Whether X model is loaded
        var model = 0f  // Model to use. Starting from 0: s,l,x
        var acc = 0f // Average accuracy

        var FPSInf = ""
        var FPSlocal = 0f // Extract only the actual FPS part from FPSInf.
        var FPSOff = "10"

        // DQN variables
        var state = floatArrayOf(h_CPU0, h_CPU4, h_GPU, h_5G, m_available, cpu0, cpu4, gpu, r, r_sent, l1, l2, model, FPSlocal, acc)
        var nextState = state
        var action = 0
        var actionToTrain = FloatArray(81)
        actionToTrain[action] = 1f
        var actionList = toBase3List(action)
        var reward = 0f
        var reward_heat = 0f
        var reward_memory = 0f
        var reward_FPS = 0f
        var reward_acc = 0f
        var rewardToTrain = FloatArray(81)
        var loss = 0f
        var flag = 1  // Prevent data learning for first action.
        rewardToTrain[action] = reward

        // Initialization
        setFreq(listOf(cpu0.toInt(), cpu4.toInt(), gpu.toInt()))
        var observation = mutableListOf(0f)
        var thermalData = listOf("H")
        var measureResult = ""
        var startMeasure = 0L
        var timeslot = 0

        // Currently set weight
        result.add(",lambda(Heat),$lambda,kappa(Memory),$kappa,mu(FPS),$mu,nu(Accuracy),$nu,nu_bias,$nuB,h_th,$h_th,m_th,$m_th,targetFPS,$targetFPS,epsilon,$epsilon,epsilonDecay,$epsilonDecay,epsilonReset,$epsilonReset,trainedModel,$trainedModel\n")
        // First column data names
        var dataName = getDataName()
        result.add(dataName)
        delay(5000)  // To remove initial FPS delay.

        if (trainedModel == 1) {
            var weights = dqnAgent.load()
            dqnAgent.restore(weights)
        }

        // Start experiment
        channelStart.send(1)
        // Initial action
        channelModel.send(listOf(l1, l2, model))
        r = off_list[r.toInt()].toFloat()
        channelR.send(r.toInt())
        setFreq(listOf(cpu0.toInt(), cpu4.toInt(), gpu.toInt()))

        while(true){
            // Receive FPS and measure
            timeslot += 1
            var startChannel2 = SystemClock.uptimeMillis()
            Log.d("ONGOING", "FPSInf receive")
            runBlocking() {
                val jobs = listOf(
                    launch {
                        FPSOff = channelOffFPS.receive()
                        Log.d("Channel", "channelFPS Off receive latency at measureCoroutine: ${SystemClock.uptimeMillis() - startChannel2}")
                        channelRChecker.send(1)
                    },
                    launch {
                        FPSInf = channelInfFPS.receive()
                        FPSlocal = FPSInf.split(",")[4].toFloat()
                        Log.d("Channel", "channelFPS Inf receive latency at measureCoroutine: ${SystemClock.uptimeMillis() - startChannel2}")
                    },
                    launch {
                        // measure.
                        Log.d("ONGOING", "getMeasure")
                        startMeasure = SystemClock.uptimeMillis()
                        measureResult = getMeasure()
                        Log.d("ONGOING", "After getMeasure")
                        thermalData = measureResult.split(",")
                        observation = mutableListOf(thermalData[0].toFloat(), thermalData[1].toFloat(), thermalData[2].toFloat(), thermalData[3].toFloat(), thermalData[6].toFloat(), thermalData[19].toFloat())  // CPU0 CPU4 GPU 5G MemAvailable
                        Log.d("Latency", "Measure latency: ${SystemClock.uptimeMillis()-startMeasure}")
                    }
                )
                jobs.forEach { it.join() }
            }
            var startElse = SystemClock.uptimeMillis()

            // Organize results.
            observation.add(FPSOff.toFloat())
            h_CPU0 = observation[0]  // current temperature of CPU
            h_CPU4 = observation[1]  // current temperature of CPU
            h_CPU7 = observation[2]  // current temperature of CPU
            h_GPU = observation[3]  // current temperature of GPU
            h_5G = observation[4]  // current temperature of 5G
            m_available = observation[5]  // current available memory
            r_sent = observation[6]  // actual number of offloaded images

            // Calculate reward.
            var startReward = SystemClock.uptimeMillis()
            var h_max = maxOf(h_CPU0, h_CPU4, h_CPU7, h_5G, h_GPU)
            var rewardList = getReward(h_max,m_available,model,FPSlocal,FPSOff.toFloat())
            reward = rewardList[0]
            reward_heat = rewardList[1]
            reward_memory = rewardList[2]
            reward_FPS = rewardList[3]
            reward_acc = rewardList[4]
            acc = rewardList[5]
            rewardToTrain = FloatArray(81)
            rewardToTrain[action] = reward



            nextState = floatArrayOf(h_CPU0, h_CPU4, h_GPU, h_5G, m_available, cpu0, cpu4, gpu, r, r_sent, l1, l2, model, FPSlocal, acc)


            // Add training sample.
            if (flag == 1) {
                flag = 0  // Pass first timeslot.
            } else {
                Log.d("ONGOING", "addSample")
                dqnAgent.addSample(state, actionToTrain, rewardToTrain, nextState)
                lrCnt += 1

                lr *= lrdecay
                if (lr < lrmin) {
                    lr = lrmin
                }

                if (lrCnt == lrResetCnt) {
                    lr = lrReset
                    lrCnt = 0
                }
                if (reward_heat < 0 || reward_memory < 0 || reward_FPS < 0) {
                    lr=lrMax
                    for (i in 0 until num_sample) {
                        dqnAgent.addSample(state, actionToTrain, rewardToTrain, nextState)
                    }
                }
                updateLR(lr)
                // latency check
                Log.d("Latency", "Reward and addSample latency: ${SystemClock.uptimeMillis()-startReward}")
            }

//            Log.d("TAG", "state: ${state.contentToString()}")
//            Log.d("TAG", "actionToTrain: ${actionToTrain.contentToString()}")
//            Log.d("TAG", "rewardToTrain: ${rewardToTrain.contentToString()}")
//            Log.d("TAG", "nextState: ${nextState.contentToString()}")

            // Save result values
            var startChannelResult = SystemClock.uptimeMillis()
            tempString = FPSInf.toString() + "," + FPSOff + "," + (FPSlocal+FPSOff.toFloat()).toString()+ ","+ acc.toString() + ","+ cpu0.toString()+","+cpu0_list[cpu0.toInt()].substring(0, cpu0_list[cpu0.toInt()].length-3)+","+cpu4.toString()+","+cpu4_list[cpu4.toInt()].substring(0,cpu4_list[cpu4.toInt()].length-3)+","+gpu.toString()+","+gpu_list[gpu.toInt()].substring(0, gpu_list[gpu.toInt()].length-3)+","+r.toString()+","+loss.toString()+","+reward_heat.toString()+","+reward_memory.toString()+","+reward_FPS.toString()+","+reward_acc.toString()+","+reward.toString()+","+lr.toString()+","+measureResult
            result.add(tempString)
            Log.d("Latency", "Add result latency: ${SystemClock.uptimeMillis() - startChannelResult}")

            // From here, next timeslot
            // Decide model loading and update target every 5 timeslots
            if (timeslot%5 == 0) {
                Log.d("ONGOING", "Model evaluation")
                channelModel.send(listOf(l1, l2, 3f))  // Temporarily run with smallest model to prevent unloading model in use.
                if (reward_memory < 0) {  // Memory shortage - memory boost
                    l2 = 0f
                    l1 = Random.nextInt(0, l1.toInt()+1).toFloat()
                    nextState[10]=l1
                    nextState[11]=l2
                } else if (reward_memory > 40) {  // Memory surplus - adaptability boost
                    l1 = Random.nextInt(l2.toInt(), 2).toFloat()
                    l2 = Random.nextInt(l2.toInt(), 2).toFloat()
                    nextState[10]=l1
                    nextState[11]=l2
                } else{
                    if (Random.nextFloat() < epsilon) {
                        l1 = Random.nextInt(0,2).toFloat()
                        l2 = Random.nextInt(0,2).toFloat()
                        nextState[10]=l1
                        nextState[11]=l2
                        Log.d("Action", "Random model selected")
                    } else {
                        nextState[10]=1f
                        nextState[11]=1f
                        val tensorBuffer = TensorBuffer.createDynamic(DataType.FLOAT32)
                        tensorBuffer.loadArray(nextState, intArrayOf(nextState.size))
                        var result11 = max(dqnAgent.predict(tensorBuffer))

                        nextState[10]=1f
                        nextState[11]=0f
                        tensorBuffer.loadArray(nextState, intArrayOf(nextState.size))
                        var result10 = max(dqnAgent.predict(tensorBuffer))

                        nextState[10]=0f
                        nextState[11]=1f
                        tensorBuffer.loadArray(nextState, intArrayOf(nextState.size))
                        var result01 = max(dqnAgent.predict(tensorBuffer))

                        nextState[10]=0f
                        nextState[11]=0f
                        tensorBuffer.loadArray(nextState, intArrayOf(nextState.size))
                        var result00 = max(dqnAgent.predict(tensorBuffer))

                        var resultMax = maxOf(result01, result10, result11, result00)
                        Log.d("ONGOING", "Model evaluation done")
                        when (resultMax) {
                            result00 -> {
                                l1 = 0f
                                l2 = 0f
                                nextState[10]=0f
                                nextState[11]=0f
                            }
                            result10 -> {
                                l1 = 1f
                                l2 = 0f
                                nextState[10]=1f
                                nextState[11]=0f
                            }
                            result01 -> {
                                l1 = 0f
                                l2 = 1f
                                nextState[10]=0f
                                nextState[11]=1f
                            }
                            result11 -> {
                                l1 = 1f
                                l2 = 1f
                                nextState[10]=1f
                                nextState[11]=1f
                            }
                        }
                    }
                }

                var startModelChange = SystemClock.uptimeMillis()


                Log.d("ONGOING", "Model load/unload prev=($l1_prev, $l2_prev), now=($l1, $l2)")
                if (l1 != l1_prev) {
                    if (l1 > l1_prev) {
                        // Load model
                        classifier_m = ClassifierWithModel(context, "yolov8m-cls_float32.tflite")
                    } else if (l1 < l1_prev) {
                        // Unload model
                        classifier_m.finish()
                    }
                }
                if (l2 != l2_prev) {
                    if (l2 > l2_prev) {
                        // Load model
                        classifier_x = ClassifierWithModel(context, "yolov8x-cls_float32.tflite")
                    } else if (l2 < l2_prev) {
                        // Unload model
                        classifier_x.finish()
                    }
                }
                Log.d("Latency", "Model load/unload latency: ${SystemClock.uptimeMillis()-startModelChange}, prev=($l1_prev, $l2_prev), now=($l1, $l2)")
                l1_prev = l1
                l2_prev = l2
                dqnAgent.update_target()
            }

            state = nextState

            // Calculate next action
            Log.d("ONGOING", "getAction")
            var startChannel3 = SystemClock.uptimeMillis()
            action = getAction(state)!!
            if (reward_heat < 0) {  // If exceeded, force it down. cool-down action
                actionList = toBase3List(action)
                var cpu0_tmp = actionList[0].toInt()
                var gpu_tmp = actionList[1].toInt()
                var r_tmp = actionList[2].toInt()
                var model_tmp = actionList[3].toInt()

                if (cpu0_tmp != 0) {
                    cpu0_tmp = Random.nextInt(0, cpu0_tmp)
                }
                if (gpu_tmp != 0) {
                    gpu_tmp = Random.nextInt(0, gpu_tmp)
                }
                if (r_tmp != 0) {
                    r_tmp = Random.nextInt(0, r_tmp)
                }

                action = fromBase3List(listOf(cpu0_tmp, gpu_tmp, r_tmp, model_tmp))
            }
            if (reward_FPS == mu) {
                if (Random.nextFloat() < 0.2) {
                    // exploration for higher accuracy  // accuracy boost action
                    Log.d("ONGOING", "")
                    var cpu0_tmp = actionList[0].toInt()
                    var gpu_tmp = actionList[1].toInt()
                    var r_tmp = actionList[2].toInt()
                    var model_tmp = actionList[3].toInt()
                    if (model_tmp != 2) {
                        model_tmp = Random.nextInt(model_tmp.toInt(), model_tmp.toInt()+2)  // Exploration only with one larger model..
                        if (model_tmp==2 && l2==0f) {
                            if (l1==1f) {
                                model_tmp = 1
                            } else {
                                model_tmp = 0
                            }
                        }
                        if (model_tmp==1 && l1==0f) {
                            if (l2 == 1f) {
                                model_tmp = 2
                            } else {
                                model_tmp = 0
                            }
                        }
                        // if model=2, no higher accuracy
                        action = fromBase3List(listOf(cpu0_tmp, gpu_tmp, r_tmp, model_tmp))
                    }
                }
            } else {
                // FPS boost action
                if (reward_heat > 0) {
                    var cpu0_tmp = actionList[0].toInt()
                    var gpu_tmp = actionList[1].toInt()
                    var r_tmp = actionList[2].toInt()
                    var model_tmp = actionList[3].toInt()
                    if (cpu0_tmp != 2) {
                        cpu0_tmp += 1
                    }
                    if (gpu_tmp != 2) {
                        gpu_tmp += 1
                    }
                    if (model_tmp == 2) {
                        if (l1 == 1f) {model_tmp = 1}
                        else {model_tmp = 0}
                    } else if (model_tmp == 1) {
                        model_tmp = 0
                    }
                    action = fromBase3List(listOf(cpu0_tmp, gpu_tmp, r_tmp, model_tmp))
                }
            }

            actionToTrain = FloatArray(81)
            actionToTrain[action] = 1f
            Log.d("Latency", "Get action latency: ${SystemClock.uptimeMillis() - startChannel3}")

            actionList = toBase3List(action)
            cpu0 = actionList[0].toFloat()
            cpu4 = actionList[0].toFloat()
            gpu = actionList[1].toFloat()
            r = actionList[2].toFloat()
            model = actionList[3].toFloat()
            Log.d("Action", "cpu0: $cpu0, cpu4: $cpu4, gpu: $gpu, r: $r, model: $model, actionNo: $action, l1: ${state[10]}, l2: ${state[11]}")
            Log.d("Latency", "Overall latency w/o FPS receive: ${SystemClock.uptimeMillis()-startElse}")

            // Execute
            var startSend = SystemClock.uptimeMillis()
            channelModel.send(listOf(l1, l2, model))
            r = off_list[r.toInt()].toFloat()
            channelR.send(r.toInt())
            Log.d("Latency", "channel Send latency in measureCoroutine: ${SystemClock.uptimeMillis()-startSend}")
            setFreq(listOf(cpu0.toInt(), cpu4.toInt(), gpu.toInt()))

            // Training
            if (dqnAgent.getTrainBatchSize() >= dqnAgent.getExpectedBatchSize()) {
                Log.d("ONGOING", "Start training")
                var startChannel4 = SystemClock.uptimeMillis()
                loss = dqnAgent.startTraining()
                Log.d("ONGOING", "Start update target")

                var endChannel4 = SystemClock.uptimeMillis() - startChannel4
                Log.d("Latency", "Training target latency: $endChannel4")
            }

            // Above is the result of previous timeslot's action. state(t-1), action(t-1), reward(t), state(t), action(t)
            // state[14]: temperature(cpu0, cpu4, gpu, 5G), available memory, model set(num_model, model),
            //        clock(cpu0, cpu4, gpu), FPS, accuracy, offloading(r, r_sent)
            // action: CPU, GPU, offloading rate, model




            // Organize results after experiment ends
            if(SystemClock.uptimeMillis() - startTime > duration*1000) {
                binding.textViewtest.text = "measurement done!"
                channelR.send(r.toInt())
                channelFinish.send(1)
                val fileName = "data_file.txt"
                val file = File(applicationContext.filesDir, fileName)
                var stringResult = result.toString()
                file.writeText(stringResult.substring(1, stringResult.length-1))
                Log.d("Finished", "File writing done!")

                // Save model
                var weightList = dqnAgent.returnWeight()
                var weightW = weightList[0]
                var weightB = weightList[1]
                // TEST FOR SAVE
                val tensorBuffer = TensorBuffer.createDynamic(DataType.FLOAT32)
                // TEST OF PREDICT
                // Loading a float array:
                val arr1 = floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f)
                Log.d("TEST", "array input: " + arr1.joinToString(","))
                tensorBuffer.loadArray(arr1, intArrayOf(arr1.size))

                var testSave = dqnAgent.predict(tensorBuffer)
                Log.d("TEST", "predict before save and load: $testSave")
                dqnAgent.save(weightW, weightB)

                dqnAgent.startTraining()
                var testSave3 = dqnAgent.predict(tensorBuffer)
                Log.d("TEST", "predict after training: $testSave3")

                var weights = dqnAgent.load()
                dqnAgent.restore(weights)
                var testSave2 = dqnAgent.predict(tensorBuffer)
                Log.d("TEST", "predict after save and load: $testSave2")
                // Convert file to content URI
                val contentUri: Uri = FileProvider.getUriForFile(applicationContext, "com.example.dualengine.fileprovider", file)

                val intent = Intent(Intent.ACTION_SEND)
                intent.type = "text/plain"
                intent.putExtra(Intent.EXTRA_STREAM, contentUri)
                intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                startActivity(Intent.createChooser(intent, "Share!"))
                delay(5000)
                job.cancel()
                break
            }
        }
        Log.d("Finished", "measureCoroutine finished")
    }

//    suspend fun CoroutineScope.freqCoroutine(channelFreq: ReceiveChannel<List<Int>>, channelFinish: ReceiveChannel<Int>){
//        // Role: Set CPU, GPU frequency. Seems unnecessary as it finishes quickly. Revive if needed later.
//        // set CPU, GPU freq
//        while(channelFinish.isEmpty){
//            var freqs = channelFreq.receive()
//            setFreq(freqs)
//        }
//    }
}

fun <T : Comparable<T>> Iterable<T>.argmax(): Int? {
    return withIndex().maxByOrNull { it.value }?.index
}
package com.example.emotiondetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.emotiondetector.ui.theme.EmotionDetectorTheme
import androidx.compose.foundation.border
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.graphics.Color as ComposeColor
import androidx.compose.ui.text.style.TextAlign
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import androidx.compose.ui.text.font.FontWeight
import android.graphics.Matrix
import android.graphics.RectF

class MainActivity : ComponentActivity() {
    private val TAG = "EmotionDetector"
    private var capturedImageBitmap by mutableStateOf<Bitmap?>(null)
    private var detectedEmotion by mutableStateOf<String?>(null)
    private var confidenceScore by mutableStateOf<Float?>(null)
    private var allEmotions by mutableStateOf<List<Pair<String, Float>>>(emptyList())
    private var tflite: Interpreter? = null
    private var isModelLoaded = false

    // List of emotion labels for the FER+ model
    private val emotionLabels = listOf("Neutral", "Happy", "Surprised", "Sad", "Angry", "Disgusted", "Fearful", "Contempt")

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            openCamera()
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
        }
    }

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        if (bitmap != null) {
            capturedImageBitmap = bitmap
            // Reset emotion detection when new image is captured
            detectedEmotion = null
            confidenceScore = null
            allEmotions = emptyList()
            Toast.makeText(this, "Image captured successfully!", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Load the TFLite model
        try {
            tflite = loadModelFile()
            isModelLoaded = true
            Log.d(TAG, "Model loaded successfully")
            Toast.makeText(this, "Model loaded successfully", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model: ${e.message}", e)
            Toast.makeText(this, "Error loading model: ${e.message}", Toast.LENGTH_LONG).show()
            isModelLoaded = false
        }

        enableEdgeToEdge()
        setContent {
            EmotionDetectorTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    EmotionDetectorApp(
                        modifier = Modifier.padding(innerPadding),
                        onCameraClick = { checkCameraPermissionAndOpen() },
                        onAnalyzeClick = { analyzeEmotion() },
                        capturedImage = capturedImageBitmap,
                        detectedEmotion = detectedEmotion,
                        confidenceScore = confidenceScore,
                        allEmotions = allEmotions,
                        isModelLoaded = isModelLoaded
                    )
                }
            }
        }
    }

    private fun loadModelFile(): Interpreter {
        try {
            val fileDescriptor = assets.openFd("fer_mobilenet.tflite")
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            // Create interpreter options to configure threading and precision
            val options = Interpreter.Options().apply {
                setNumThreads(4) // Use 4 threads for faster inference
            }

            val interpreter = Interpreter(mappedByteBuffer, options)
            fileDescriptor.close()
            inputStream.close()

            return interpreter
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            throw e
        }
    }

    private fun checkCameraPermissionAndOpen() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                openCamera()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun openCamera() {
        takePictureLauncher.launch(null)
    }

    private fun analyzeEmotion() {
        if (!isModelLoaded) {
            Toast.makeText(this, "Model not loaded properly", Toast.LENGTH_SHORT).show()
            return
        }

        capturedImageBitmap?.let { bitmap ->
            try {
                // Get model input shape
                val inputShape = tflite?.getInputTensor(0)?.shape()
                val inputHeight = inputShape?.get(1) ?: 224
                val inputWidth = inputShape?.get(2) ?: 224

                Log.d(TAG, "Model input shape: $inputHeight x $inputWidth")

                // Preprocess the image for the model
                val processedBitmap = preprocessImage(bitmap, inputWidth, inputHeight)

                // Convert bitmap to ByteBuffer
                val inputBuffer = convertBitmapToByteBuffer(processedBitmap)

                // Create output buffer - shape will depend on your model
                val outputShape = tflite?.getOutputTensor(0)?.shape()
                val outputSize = outputShape?.get(1) ?: emotionLabels.size

                Log.d(TAG, "Model output size: $outputSize")

                val outputBuffer = Array(1) { FloatArray(outputSize) }

                // Run inference
                Log.d(TAG, "Running inference")
                tflite?.run(inputBuffer, outputBuffer)

                // Get results
                val result = outputBuffer[0]

                // Log results for debugging
                emotionLabels.forEachIndexed { index, label ->
                    if (index < result.size) {
                        Log.d(TAG, "$label: ${result[index]}")
                    }
                }

                // Store all emotion probabilities
                val emotionProbabilities = emotionLabels.mapIndexed { index, label ->
                    if (index < result.size) {
                        Pair(label, result[index])
                    } else {
                        Pair(label, 0f)
                    }
                }.sortedByDescending { it.second }

                allEmotions = emotionProbabilities

                // Find the emotion with highest probability
                if (emotionProbabilities.isNotEmpty()) {
                    val topEmotion = emotionProbabilities.first()
                    detectedEmotion = topEmotion.first
                    confidenceScore = topEmotion.second

                    Log.d(TAG, "Detected emotion: $detectedEmotion with confidence: $confidenceScore")
                    Toast.makeText(this, "Detected: $detectedEmotion", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error analyzing emotion", e)
                Toast.makeText(this, "Error analyzing emotion: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun preprocessImage(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        try {
            // 1. Calculate crop area to maintain aspect ratio
            val sourceWidth = bitmap.width
            val sourceHeight = bitmap.height

            // Compute the square center crop
            val xOffset: Int
            val yOffset: Int
            val scaleFactor: Float

            if (sourceWidth > sourceHeight) {
                scaleFactor = targetHeight.toFloat() / sourceHeight
                xOffset = (sourceWidth - sourceHeight) / 2
                yOffset = 0
            } else {
                scaleFactor = targetWidth.toFloat() / sourceWidth
                xOffset = 0
                yOffset = (sourceHeight - sourceWidth) / 2
            }

            // Define crop and destination rectangles
            val sourceCrop = RectF(
                xOffset.toFloat(),
                yOffset.toFloat(),
                (sourceWidth - xOffset).toFloat(),
                (sourceHeight - yOffset).toFloat()
            )

            val destRect = RectF(0f, 0f, targetWidth.toFloat(), targetHeight.toFloat())

            // Create new bitmap of desired size
            val scaledBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)

            // Draw source bitmap to destination bitmap using transformation matrix
            val canvas = android.graphics.Canvas(scaledBitmap)
            val matrix = Matrix()
            matrix.setRectToRect(sourceCrop, destRect, Matrix.ScaleToFit.FILL)
            canvas.drawBitmap(bitmap, matrix, null)

            return scaledBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Error preprocessing image", e)
            throw e
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val inputSize = bitmap.width // Assuming square image after preprocessing

        // For RGB - 3 channels, 4 bytes per channel (float32)
        val modelInputSize = 4 * inputSize * inputSize * 3

        val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // Reset position to beginning
        byteBuffer.rewind()

        for (pixel in pixels) {
            // Extract RGB values
            val r = Color.red(pixel)
            val g = Color.green(pixel)
            val b = Color.blue(pixel)

            // Normalize pixel values to [-1,1]
            // MobileNet models typically use this normalization
            val normalizedR = (r - 127.5f) / 127.5f
            val normalizedG = (g - 127.5f) / 127.5f
            val normalizedB = (b - 127.5f) / 127.5f

            // Add RGB channels
            byteBuffer.putFloat(normalizedR)
            byteBuffer.putFloat(normalizedG)
            byteBuffer.putFloat(normalizedB)
        }

        // Reset position to beginning before returning
        byteBuffer.rewind()
        return byteBuffer
    }

    override fun onDestroy() {
        super.onDestroy()
        // Close the interpreter when the activity is destroyed
        tflite?.close()
    }
}

@Composable
fun EmotionDetectorApp(
    modifier: Modifier = Modifier,
    onCameraClick: () -> Unit,
    onAnalyzeClick: () -> Unit,
    capturedImage: Bitmap?,
    detectedEmotion: String?,
    confidenceScore: Float?,
    allEmotions: List<Pair<String, Float>>,
    isModelLoaded: Boolean
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "Emotion Detector",
            style = MaterialTheme.typography.headlineMedium,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )

        if (!isModelLoaded) {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = "Model not loaded properly. Please restart the app.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    modifier = Modifier.padding(16.dp),
                    textAlign = TextAlign.Center
                )
            }
        }

        Box(
            modifier = Modifier
                .size(250.dp)
                .border(2.dp, ComposeColor.Gray, RoundedCornerShape(8.dp)),
            contentAlignment = Alignment.Center
        ) {
            if (capturedImage != null) {
                Image(
                    bitmap = capturedImage.asImageBitmap(),
                    contentDescription = "Captured Image",
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Text("No image captured yet")
            }
        }

        // Display detected emotion
        if (detectedEmotion != null) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp)
            ) {
                Column(
                    modifier = Modifier
                        .padding(16.dp)
                        .fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Primary Emotion",
                        style = MaterialTheme.typography.titleMedium
                    )

                    Text(
                        text = detectedEmotion,
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )

                    confidenceScore?.let {
                        Text(
                            text = "Confidence: ${(it * 100).toInt()}%",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }

            // Display all emotions with progress bars
            if (allEmotions.isNotEmpty()) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .padding(16.dp)
                            .fillMaxWidth()
                    ) {
                        Text(
                            text = "All Emotions",
                            style = MaterialTheme.typography.titleMedium,
                            modifier = Modifier.padding(bottom = 8.dp)
                        )

                        // Show top 3 emotions with progress bars
                        allEmotions.take(3).forEach { (emotion, score) ->
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 4.dp)
                            ) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = emotion,
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Text(
                                        text = "${(score * 100).toInt()}%",
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                }

                                LinearProgressIndicator(
                                    progress = score,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(8.dp)
                                        .padding(top = 4.dp)
                                )
                            }
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.weight(1f))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = onCameraClick,
                modifier = Modifier
                    .weight(1f)
                    .height(56.dp)
            ) {
                Text("Take Photo")
            }

            // Analyze emotion button
            if (capturedImage != null) {
                Button(
                    onClick = onAnalyzeClick,
                    modifier = Modifier
                        .weight(1f)
                        .height(56.dp),
                    enabled = isModelLoaded
                ) {
                    Text("Analyze Emotion")
                }
            }
        }
    }
}
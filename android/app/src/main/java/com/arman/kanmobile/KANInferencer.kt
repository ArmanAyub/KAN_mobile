package com.arman.kanmobile

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

data class InferenceResult(
    val label: String,
    val confidence: Float,
    val hiddenActivations: FloatArray  // 64-dim KAN hidden layer output
)

class KANInferencer(context: Context) {

    private val env     = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // ImageNet normalization constants
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

    init {
        val modelBytes = context.assets.open("kan_model_android.onnx").readBytes()
        session = env.createSession(modelBytes, OrtSession.SessionOptions())
    }

    fun infer(bitmap: Bitmap): InferenceResult {
        val resized    = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputTensor = bitmapToTensor(resized)

        val inputs  = mapOf("input" to inputTensor)
        val outputs = session.run(inputs)

        // Output 0: logits [1, 2]
        val logits = (outputs[0].value as Array<FloatArray>)[0]
        // Output 1: hidden activations [1, 64]
        val hidden = (outputs[1].value as Array<FloatArray>)[0]

        val probs   = softmax(logits)
        val predIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        val label   = if (predIdx == 0) "HUMAN" else "NON-HUMAN"

        inputTensor.close()
        outputs.close()

        return InferenceResult(label, probs[predIdx], hidden)
    }

    private fun bitmapToTensor(bitmap: Bitmap): OnnxTensor {
        val pixels = IntArray(224 * 224)
        bitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)

        val buffer = FloatBuffer.allocate(1 * 3 * 224 * 224)
        val rChannel = FloatArray(224 * 224)
        val gChannel = FloatArray(224 * 224)
        val bChannel = FloatArray(224 * 224)

        for (i in pixels.indices) {
            val px = pixels[i]
            rChannel[i] = (((px shr 16) and 0xFF) / 255f - mean[0]) / std[0]
            gChannel[i] = (((px shr 8)  and 0xFF) / 255f - mean[1]) / std[1]
            bChannel[i] = (((px)        and 0xFF) / 255f - mean[2]) / std[2]
        }

        buffer.put(rChannel)
        buffer.put(gChannel)
        buffer.put(bChannel)
        buffer.rewind()

        return OnnxTensor.createTensor(env, buffer, longArrayOf(1, 3, 224, 224))
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max  = logits.max()
        val exps = logits.map { Math.exp((it - max).toDouble()).toFloat() }.toFloatArray()
        val sum  = exps.sum()
        return exps.map { it / sum }.toFloatArray()
    }

    fun close() {
        session.close()
        env.close()
    }
}

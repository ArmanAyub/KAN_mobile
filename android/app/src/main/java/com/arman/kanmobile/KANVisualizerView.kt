package com.arman.kanmobile

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class KANVisualizerView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val NUM_BARS = 64
    private val BAR_RADIUS = 4f
    private val BAR_GAP_RATIO = 0.25f   // gap as fraction of bar width

    // Smoothed activation values (lerp toward target each frame)
    private val displayValues  = FloatArray(NUM_BARS) { 0f }
    private val targetValues   = FloatArray(NUM_BARS) { 0f }
    private val LERP_FACTOR    = 0.25f

    // Human = blue, Non-Human = red, interpolated by confidence
    private var humanWeight = 0.5f   // 0.0 = full red, 1.0 = full blue

    private val humanColorTop    = Color.parseColor("#FF2196F3")
    private val humanColorBottom = Color.parseColor("#FF0D47A1")
    private val redColorTop      = Color.parseColor("#FFF44336")
    private val redColorBottom   = Color.parseColor("#FFB71C1C")

    private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val bgPaint  = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1AFFFFFF")
        style = Paint.Style.FILL
    }

    private val updateRunnable = object : Runnable {
        override fun run() {
            var changed = false
            for (i in 0 until NUM_BARS) {
                val prev = displayValues[i]
                displayValues[i] = prev + (targetValues[i] - prev) * LERP_FACTOR
                if (Math.abs(displayValues[i] - targetValues[i]) > 0.001f) changed = true
            }
            invalidate()
            if (changed) postDelayed(this, 16L)  // ~60fps
        }
    }

    fun update(activations: FloatArray, humanConfidence: Float) {
        // Normalize activations to [0, 1] using min-max
        val min = activations.min()
        val max = activations.max()
        val range = (max - min).coerceAtLeast(1e-6f)
        for (i in 0 until NUM_BARS.coerceAtMost(activations.size)) {
            targetValues[i] = (activations[i] - min) / range
        }
        humanWeight = humanConfidence
        removeCallbacks(updateRunnable)
        post(updateRunnable)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w == 0f || h == 0f) return

        val totalGap  = w * BAR_GAP_RATIO
        val barWidth  = (w - totalGap) / NUM_BARS
        val gapWidth  = totalGap / (NUM_BARS + 1)

        for (i in 0 until NUM_BARS) {
            val barH   = (displayValues[i] * (h - 8f)).coerceAtLeast(4f)
            val left   = gapWidth + i * (barWidth + gapWidth)
            val right  = left + barWidth
            val top    = h - barH
            val bottom = h

            // Interpolate color between human (blue) and non-human (red)
            val topColor    = blendColors(redColorTop,    humanColorTop,    humanWeight)
            val bottomColor = blendColors(redColorBottom, humanColorBottom, humanWeight)

            val gradient = LinearGradient(
                left, top, left, bottom,
                topColor, bottomColor,
                Shader.TileMode.CLAMP
            )
            barPaint.shader = gradient

            val rect = RectF(left, top, right, bottom)
            canvas.drawRoundRect(rect, BAR_RADIUS, BAR_RADIUS, barPaint)
        }
    }

    private fun blendColors(c1: Int, c2: Int, ratio: Float): Int {
        val r = ratio.coerceIn(0f, 1f)
        val a = Color.alpha(c1) + ((Color.alpha(c2) - Color.alpha(c1)) * r).toInt()
        val red = Color.red(c1) + ((Color.red(c2) - Color.red(c1)) * r).toInt()
        val g = Color.green(c1) + ((Color.green(c2) - Color.green(c1)) * r).toInt()
        val b = Color.blue(c1) + ((Color.blue(c2) - Color.blue(c1)) * r).toInt()
        return Color.argb(a, red, g, b)
    }
}

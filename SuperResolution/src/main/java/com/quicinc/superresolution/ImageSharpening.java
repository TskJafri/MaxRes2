package com.quicinc.superresolution;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.util.Pair;

import com.quicinc.tflite.AIHubDefaults;
import com.quicinc.tflite.TFLiteHelpers;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.Map;

public class ImageSharpening implements AutoCloseable {
    private static final String TAG = "ImageSharpening";
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;
    private final int[] inputShape;
    private final DataType inputType;
    private final DataType outputType;
    private long preprocessingTime;
    private long postprocessingTime;
    private final ImageProcessor inputImageProcessor;
    private final ImageProcessor outputImageProcessor;
    private final TensorBuffer outputBuffer;
    private final Map<Integer, Object> outputBindings;
    private final TensorImage outputImage;

    public ImageSharpening(Context context, String modelPath) throws IOException, NoSuchAlgorithmException {
        this(context, modelPath, AIHubDefaults.delegatePriorityOrder);
    }

    public ImageSharpening(Context context, String modelPath, TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {
        // Load TF Lite model
        Pair<MappedByteBuffer, String> modelAndHash = TFLiteHelpers.loadModelFile(context.getAssets(), modelPath);
        Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
                modelAndHash.first,
                delegatePriorityOrder,
                AIHubDefaults.numCPUThreads,
                context.getApplicationInfo().nativeLibraryDir,
                context.getCacheDir().getAbsolutePath(),
                modelAndHash.second
        );
        tfLiteInterpreter = iResult.first;
        tfLiteDelegateStore = iResult.second;

        // Validate TF Lite model fits requirements for this app
        assert tfLiteInterpreter.getInputTensorCount() == 1 : "Expected 1 input tensor, but got " + tfLiteInterpreter.getInputTensorCount();
        Tensor inputTensor = tfLiteInterpreter.getInputTensor(0);
        inputShape = inputTensor.shape();
        inputType = inputTensor.dataType();
        assert inputShape.length == 4 : "Expected input tensor to be 4D, but got " + inputShape.length;
        assert inputShape[0] == 1 : "Expected batch size of 1, but got " + inputShape[0];
        assert inputShape[3] == 3 : "Expected 3 channels, but got " + inputShape[3];
        assert inputType == DataType.UINT8 || inputType == DataType.FLOAT32 : "Expected input type UINT8 or FLOAT32, but got " + inputType;

        assert tfLiteInterpreter.getOutputTensorCount() == 1 : "Expected 1 output tensor, but got " + tfLiteInterpreter.getOutputTensorCount();
        Tensor outputTensor = tfLiteInterpreter.getOutputTensor(0);
        int[] outputShape = outputTensor.shape();
        outputType = outputTensor.dataType();
        assert outputShape.length == 4 : "Expected output tensor to be 4D, but got " + outputShape.length;
        assert outputShape[0] == 1 : "Expected batch size of 1, but got " + outputShape[0];
        Log.d(TAG, "Input Shape: " + inputShape[1] + "x" + inputShape[2]);
        Log.d(TAG, "Output Shape: " + outputShape[1] + "x" + outputShape[2]);
        if (outputShape[1] != inputShape[1] || outputShape[2] != inputShape[2]) {
            Log.w(TAG, "Output shape does not match input shape. Adjusting postprocessing.");
        }
        assert outputShape[3] == 3 : "Expected 3 channels, but got " + outputShape[3];
        assert outputType == DataType.UINT8 || inputType == DataType.FLOAT32 : "Expected output type UINT8 or FLOAT32, but got " + outputType;

        // Set-up preprocessor
        inputImageProcessor = new ImageProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build();
        outputImageProcessor = new ImageProcessor.Builder().add(new NormalizeOp(0.0f, 1 / 255.0f)).build();

        // Set-up output image
        outputBuffer = TensorBuffer.createFixedSize(outputShape, outputType);
        outputBindings = new HashMap<>();
        outputBindings.put(0, outputBuffer.getBuffer());
        outputImage = new TensorImage(outputType);
        outputImage.load(outputBuffer);
    }

    @Override
    public void close() {
        tfLiteInterpreter.close();
        for (Delegate delegate : tfLiteDelegateStore.values()) {
            delegate.close();
        }
    }

    private ByteBuffer[] preprocess(Bitmap image) {
        long prepStartTime = System.nanoTime();

        // Convert type and fill input buffer
        ByteBuffer inputBuffer;
        TensorImage tImg = TensorImage.fromBitmap(image);
        if (inputType == DataType.FLOAT32) {
            // Divide float values by 255
            inputBuffer = inputImageProcessor.process(tImg).getBuffer();
        } else {
            inputBuffer = tImg.getTensorBuffer().getBuffer();
        }

        preprocessingTime = System.nanoTime() - prepStartTime;
        Log.d(TAG, "Preprocessing Time: " + preprocessingTime / 1000000 + " ms");

        return new ByteBuffer[]{inputBuffer};
    }

    private Bitmap postprocess() {
        long postStartTime = System.nanoTime();

        TensorImage img = outputImage;
        if (outputType == DataType.FLOAT32) {
            // Multiply float values by 255
            img = outputImageProcessor.process(outputImage);
        }
        Bitmap bitmap = img.getBitmap();

        postprocessingTime = System.nanoTime() - postStartTime;
        Log.d(TAG, "Postprocessing Time: " + postprocessingTime / 1000000 + " ms");

        return bitmap;
    }

    public Bitmap generateSharpenedImage(Bitmap image) {
        // Preprocessing: convert type
        ByteBuffer[] inputs = preprocess(image);

        // Inference
        outputBuffer.getBuffer().clear();
        tfLiteInterpreter.runForMultipleInputsOutputs(inputs, outputBindings);

        // Postprocessing: Convert to bitmap
        return postprocess();
    }

    public long getLastPreprocessingTime() {
        if (preprocessingTime == 0) {
            throw new RuntimeException("Cannot get preprocessing time as model has not yet been executed.");
        }
        return preprocessingTime;
    }

    public long getLastInferenceTime() {
        return tfLiteInterpreter.getLastNativeInferenceDurationNanoseconds();
    }

    public long getLastPostprocessingTime() {
        if (postprocessingTime == 0) {
            throw new RuntimeException("Cannot get postprocessing time as model has not yet been executed.");
        }
        return postprocessingTime;
    }

    public int[] getInputWidthHeight() {
        return new int[]{inputShape[1], inputShape[2]};
    }
}

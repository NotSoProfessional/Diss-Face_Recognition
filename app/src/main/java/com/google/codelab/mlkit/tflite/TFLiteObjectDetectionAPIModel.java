/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.codelab.mlkit.tflite;

import static org.opencv.ml.Ml.ROW_SAMPLE;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;
import android.util.Pair;

import com.google.codelab.mlkit.env.Logger;

import org.apache.commons.math3.linear.MatrixUtils;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.TrainData;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.*;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.model.Model;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 *
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class TFLiteObjectDetectionAPIModel
        implements SimilarityClassifier {

  private static final Logger LOGGER = new Logger();

  //private static final int OUTPUT_SIZE = 512;
  private static final int OUTPUT_SIZE = 192;

  // Only return this many results.
  private static final int NUM_DETECTIONS = 1;

  // Float model
  private static final float IMAGE_MEAN = 128.0f;
  private static final float IMAGE_STD = 128.0f;

  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes
  private float[] numDetections;

  private AssetManager assetManager;
  private String modelFilename;
  private boolean useGPU;

  private static float[][] embeedings;

  private ByteBuffer imgData;

  private Interpreter tfLite;


  private static LinkedHashMap<String, Recognition> registered = new LinkedHashMap<>();
  private static Mat points = new Mat();
  private static Mat names = new Mat();
  //private  ArrayList<String> nams = new ArrayList<>();

  private long predict = 0;

  public void register(String name, Recognition rec) {
      registered.put(name, rec);
    //((Map.Entry<String, Recognition>) registered).getValue().get

      //nams.add(name);

      float[] data = ((float[][]) rec.getExtra())[0];

      Mat temp = new Mat(1,192, CvType.CV_32FC1);
      temp.put(0,0,data);

      points.push_back(temp);

      temp = new Mat(1,1, CvType.CV_32FC1);
      temp.put(0,0, registered.entrySet().size()-1);

      names.push_back(temp);
  }

  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static SimilarityClassifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final boolean isQuantized,
      final boolean useNNAPI,
      final boolean useGPU,
      final boolean useXXNPack,
      final int numThreads)
      throws IOException {

    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    InputStream labelsInput = assetManager.open(actualFilename);
    BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    d.inputSize = inputSize;

    d.InitModel(assetManager, modelFilename, useNNAPI, useGPU, useXXNPack, numThreads);
    d.modelFilename = modelFilename;
    d.assetManager = assetManager;

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    //d.tfLite.setNumThreads(NUM_THREADS);
    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];

    return d;
  }

  public void InitModel(final AssetManager assetManager,
                         final String modelFilename,
                         final boolean useNNAPI,
                         final boolean useGPU,
                        final boolean useXXNPACK,
                        int numThreads){
    Interpreter.Options opts = new Interpreter.Options();


    CompatibilityList compatList = new CompatibilityList();

      if (compatList.isDelegateSupportedOnThisDevice() & useGPU) {
        // If the device has a supported GPU, add the GPU delegate
        opts.setUseNNAPI(false);

        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();

        delegateOptions.setPrecisionLossAllowed(true);

        GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        opts.addDelegate(gpuDelegate);

        Log.d("NSP debug", "Using GPU!");
      } else {
        // If the GPU is not supported, run on CPU
        opts.setUseNNAPI(useNNAPI);
        opts.setUseXNNPACK(useXXNPACK);
        opts.setNumThreads(numThreads);
        Log.d("NSP debug", "Using NNAPI: " + useNNAPI + " Using XNNPack: " + useXXNPACK);
      }



    try {
      tfLite = new Interpreter(loadModelFile(assetManager, modelFilename), opts);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  // looks for the nearest embeeding in the dataset (using L2 norm)
  // and retrurns the pair <id, distance>
  private Pair<String, Float> findNearest(float[] emb) {

    long startTime = System.nanoTime();


boolean useCV = false;
    Pair<String, Float> ret = null;

    if(useCV){

      KNearest knn = KNearest.create();
      knn.setAlgorithmType(KNearest.BRUTE_FORCE);
      knn.train(points, ROW_SAMPLE, names);

      Mat target = new Mat(1, 192, CvType.CV_32FC1);
      target.put(0,0, emb);

      Mat results = new Mat();
      Mat neighborResponses = new Mat();
      Mat dists = new Mat();

      knn.findNearest(target, 1, results, neighborResponses, dists);
      int idx = (int) results.get(0,0)[0];
      //ret = new Pair<>(nams.get(idx), (float) dists.get(0,0)[0]);
      ret = new Pair<>(registered.keySet().toArray()[idx].toString(), (float) dists.get(0,0)[0]);
      System.out.println("Result: " + Integer.toString((int) results.get(0,0)[0]));
    }

  if (!useCV){
    for (Map.Entry<String, Recognition> entry : registered.entrySet()) {
      final String name = entry.getKey();
      final float[] knownEmb = ((float[][]) entry.getValue().getExtra())[0];

      float distance = 0;
      for (int i = 0; i < emb.length; i++) {
        float diff = emb[i] - knownEmb[i];
        distance += diff*diff;
      }
      distance = (float) Math.sqrt(distance);
      if (ret == null || distance < ret.second) {
        ret = new Pair<>(name, distance);
      }
    }
  }


    long endTime = System.nanoTime();

    long elapsedTime = endTime - startTime;

    System.out.println("Prediction time: " + elapsedTime + " [ns]");
    predict = elapsedTime;


    return ret;

  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap, boolean storeExtra) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    //if (useGPU){
    //  InitModel(assetManager, modelFilename, false, true, false);
    //}

    Trace.beginSection("normaliseBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");


    Object[] inputArray = {imgData};

    Trace.endSection();

// Here outputMap is changed to fit the Face Mask detector
    Map<Integer, Object> outputMap = new HashMap<>();

    embeedings = new float[1][OUTPUT_SIZE];
    outputMap.put(0, embeedings);


    // Run the inference call.
    Trace.beginSection("run");
    //tfLite.runForMultipleInputsOutputs(inputArray, outputMapBack);
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();


    float distance = Float.MAX_VALUE;
    String id = "0";
    String label = "?";

    if (registered.size() > 0) {
        //LOGGER.i("dataset SIZE: " + registered.size());
        final Pair<String, Float> nearest = findNearest(embeedings[0]);
        if (nearest != null) {

            final String name = nearest.first;
            label = name;
            distance = nearest.second;

            LOGGER.i("nearest: " + name + " - distance: " + distance);


        }
    }


    final int numDetectionsOutput = 1;
    final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
    Recognition rec = new Recognition(
            id,
            label,
            distance,
            new RectF());

    recognitions.add( rec );

    if (storeExtra) {
        rec.setExtra(embeedings);
    }

    Trace.endSection();
    return recognitions;

  }

  @Override
  public void enableStatLogging(final boolean logStats) {}


  @Override
  public void close() {}

  public void ReInitModel(final AssetManager assetManager,
                        final String modelFilename,
                        final boolean useNNAPI,
                        final boolean useGPU,
                          final boolean useXNNPack,
                          int numThreads) {
    this.useGPU = useGPU;
    InitModel(assetManager, modelFilename, useNNAPI, useGPU, useXNNPack, numThreads);
  }


  public Long getInferenceTime(){
    return tfLite.getLastNativeInferenceDurationNanoseconds();
  }

  public Long getPredictTime(){
    return predict;
  }

  public class Options{

  }
}

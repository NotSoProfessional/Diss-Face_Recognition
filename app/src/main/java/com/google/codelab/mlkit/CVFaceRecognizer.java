package com.google.codelab.mlkit;

import static org.bytedeco.opencv.global.opencv_imgproc.CV_RGB2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_RGBA2GRAY;
import static org.opencv.core.CvType.CV_32SC1;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.core.CvType.CV_8UC4;
import static org.opencv.imgproc.Imgproc.cvtColor;

import android.graphics.Bitmap;
import android.util.Log;

//import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Range;
import org.opencv.core.Size;
import org.opencv.face.EigenFaceRecognizer;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.FisherFaceRecognizer;
import org.opencv.face.LBPHFaceRecognizer;

import java.util.ArrayList;
import java.util.List;

public class CVFaceRecognizer{

    enum Method{
        FISHERFACE,
        EIGENFACE,
        LBPH
    }

    private FaceRecognizer fr;
    private Method method;
    private static ArrayList<Mat> faces = new ArrayList<>();
    private static ArrayList<Mat> new_faces  = new ArrayList<>();
    private static Mat labels = new Mat();
    private static Mat new_labels = new Mat();

    private long predictTime = 0;
    private long trainTime = 0;

    public CVFaceRecognizer(Method method){
        this.method = method;

        switch (method){
            case FISHERFACE:
                fr = FisherFaceRecognizer.create();
                break;
            case EIGENFACE:
                fr = EigenFaceRecognizer.create();
                break;
            case LBPH:
                fr = LBPHFaceRecognizer.create();
                break;
        }
    }

    public Result predict(Mat face){
        int[] label = new int[1];
        double[] confidence = new double[1];

        cvtColor(face, face, CV_RGBA2GRAY);

        long startTime = System.nanoTime();

        fr.predict(face, label, confidence);

        long endTime = System.nanoTime();
        long elapsedTime = endTime - startTime;

        predictTime = elapsedTime;

        Log.d("NSP debug", "CV Prediction time: " + elapsedTime + " [ns]");

        return new Result(label[0], confidence[0]);
    }

    public Result predict(Bitmap face){
        Mat mat = new Mat(112, 112, CV_8UC1);

        Utils.bitmapToMat(face, mat);

        return predict(mat);
    }

    public long getTrainTime() { return trainTime;}
    public long getPredictTime() { return predictTime;}

    public void train(){
        long startTime = System.nanoTime();

        if (method == Method.LBPH){
            fr.update(new_faces, new_labels);

            new_faces.clear();
            new_labels = new Mat();
        }else{
            /*Log.d("NSP debug", "Number of faces: " + faces.size());

            for (int i = 0; i < faces.size(); i++) {
                Mat face = faces.get(i);
                System.out.println("Face " + i + ": rows = " + face.rows() + ", cols = " + face.cols() + ", type = " + CvType.typeToString(face.type()) + " (" + face.type() + ")");
            }

            for (int i = 0; i < labels.rows(); i++) {
                int label = (int) labels.get(i, 0)[0];
                System.out.println("Label " + i + ": " + label);
            }

            System.out.println("Number of labels: " + labels.rows());*/

            fr.train(faces, labels);
        }

        long endTime = System.nanoTime();
        long elapsedTime = endTime - startTime;

        trainTime = elapsedTime;

        Log.d("NSP debug", "Training time: " + elapsedTime + " [ns]");
    }

    public void clearFaces(){
        faces.clear();
        labels = new Mat();
    }

    public void addFace(Mat face, Mat label){
        cvtColor(face, face, CV_RGB2GRAY);

        faces.add(face);
        labels.push_back(label);

        if (method == Method.LBPH){
            new_faces.add(face);
            new_labels.push_back(label);
        }
    }

    public void addFace(Mat face, Integer label){
        Mat mat = new Mat(1,1, CV_32SC1);
        mat.put(0,0, label);

        addFace(face, mat);
    }

    public void addFace(Bitmap face, Integer label){
        Mat mat = new Mat();

        Utils.bitmapToMat(face, mat);

        addFace(mat, label);
    }

    public void addFacesAsMat(List<Mat> faces, List<Integer> labels){
        for (int i = 0; i < faces.size(); i++) addFace(faces.get(i), labels.get(i));
    }

    public void addFacesAsBmp(List<Bitmap> faces, List<Integer> labels){
        for (int i = 0; i < faces.size(); i++) addFace(faces.get(i), labels.get(i));
    }

    public class Result {
        int label;
        double confidence;

        public Result(int label, double confidence){
            this.label = label;
            this.confidence = confidence;
        }
    }

    public class Options {

    }
}

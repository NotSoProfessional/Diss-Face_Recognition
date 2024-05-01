package com.google.codelab.mlkit;

import static org.bytedeco.opencv.global.opencv_imgproc.CV_RGBA2RGB;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgproc.Imgproc.cvtColor;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.os.Trace;

import com.google.mlkit.vision.face.Face;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import androidx.renderscript.Allocation;
import androidx.renderscript.Element;
import androidx.renderscript.RenderScript;
import androidx.renderscript.ScriptIntrinsic;
import androidx.renderscript.ScriptIntrinsicBlur;

import androidx.renderscript.ScriptIntrinsicColorMatrix;
import androidx.renderscript.ScriptIntrinsicHistogram;
import androidx.renderscript.ScriptIntrinsicLUT;


public class ImagePreProcessor {

    private Options options = new Options();

    private long alignTime = 0;
    private long equaliseTime = 0;
    private long grayTime = 0;
    private long bilateralTime = 0;

    private Context mContext;

    public ImagePreProcessor(){}

    public ImagePreProcessor(Options options, Context context){
        this.options = options;
        mContext = context;
    }

    public Bitmap Process(Face face, Bitmap input){

        Trace.beginSection("Image Pre-Processing");

        int width = face.getBoundingBox().width();
        int height = face.getBoundingBox().height();
        float x = face.getBoundingBox().centerX();
        float y = face.getBoundingBox().centerY();
        float xOffset = width / 2.0f;
        float yOffset = height / 2.0f;
        float left = x - xOffset;
        float top = y - yOffset;

        if (left < 0) left = 0;
        if (top < 0) top = 0;
        if (left + width > input.getWidth()) width -= (left + width) - input.getWidth();
        if (top + height > input.getHeight()) height -= (top + height) - input.getHeight();

        Bitmap processed = Bitmap.createBitmap(input);

        Mat mat = new Mat();

        //---------------------
        // Align
        //---------------------

        Trace.beginSection("Alignment");

        long startTime = System.nanoTime();

        if (options.rotate & options.useOpenCV){
            Utils.bitmapToMat(processed, mat);
            Point rotPoint = new Point(x, y);
            Mat rotMat  = Imgproc.getRotationMatrix2D(rotPoint, -face.getHeadEulerAngleZ(), 1);
            Imgproc.warpAffine(mat, mat, rotMat, mat.size());

            Utils.matToBitmap(mat, processed);
        }

        if (options.rotate & !options.useOpenCV){

            Bitmap rotated = Bitmap.createBitmap(processed.getWidth(), processed.getHeight(), Bitmap.Config.ARGB_8888);

            Canvas canvas = new Canvas(rotated);
            Paint drawPaint = new Paint();
            drawPaint.setAntiAlias(false);
            drawPaint.setFilterBitmap(false);

            Matrix matrix = new Matrix();
            matrix.setRotate(face.getHeadEulerAngleZ(), (int) x, (int) y);
            canvas.setMatrix(matrix);

            canvas.drawBitmap(processed, 0,0, drawPaint);

            processed = rotated;

        }

        // Crop image to face
        if (!options.useOpenCV){
            processed = Bitmap.createBitmap(processed, (int) left, (int) top, width, height);
        }

        if (options.useOpenCV){
            Utils.bitmapToMat(processed, mat);

            Rect face_region = new Rect((int) left, (int) top, width, height);
            mat = new Mat(mat, face_region);

            processed = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888);

            Utils.matToBitmap(mat, processed);
        }

        long endTime = System.nanoTime();
        alignTime = endTime - startTime;
        Trace.endSection();

        //---------------------
        // Grayscale
        //---------------------

        Trace.beginSection("RGB2Gray");
        startTime = System.nanoTime();

        if (options.grayscale & options.useOpenCV) {
            Utils.bitmapToMat(processed, mat);

            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2GRAY);

            Utils.matToBitmap(mat, processed);
        }

        if (options.grayscale & !options.useOpenCV) {
            Bitmap bm_grayscale = Bitmap.createBitmap(processed.getWidth(), processed.getHeight(), Bitmap.Config.ARGB_8888);

            Canvas canvas = new Canvas(bm_grayscale);
            Paint paint = new Paint();
            ColorMatrix colorMatrix = new ColorMatrix();
            colorMatrix.setSaturation(0); // Set saturation to 0 to convert to grayscale
            paint.setColorFilter(new ColorMatrixColorFilter(colorMatrix));

            canvas.drawBitmap(processed, 0, 0, paint);

            processed = bm_grayscale;
        }
        endTime = System.nanoTime();
        grayTime = endTime - startTime;
        Trace.endSection();


        //---------------------
        // Equalise
        //---------------------
        Trace.beginSection("Equalise");

        startTime = System.nanoTime();

        if (options.useOpenCV & options.equalise & !options.grayscale){
            Utils.bitmapToMat(processed, mat);
            Mat lum = new Mat();
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2YUV);
            Core.extractChannel(mat,lum, 0);
            Imgproc.equalizeHist(lum,lum);
            Core.insertChannel(lum,mat,0);
            Imgproc.cvtColor(mat,mat, Imgproc.COLOR_YUV2RGB);
            Utils.matToBitmap(mat, processed);
        }

        if (options.equalise & options.useOpenCV & options.grayscale){
            Utils.bitmapToMat(processed,mat);
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2GRAY);
            Imgproc.equalizeHist(mat, mat);
            Utils.matToBitmap(mat, processed);
        }

        if (options.equalise & !options.useOpenCV){

            // Compute the histogram of the grayscale image
            int[] histogram = new int[256];
            for (int x_i = 0; x_i < processed.getWidth(); x_i++) {
                for (int y_i = 0; y_i < processed.getHeight(); y_i++) {
                    int pixel = processed.getPixel(x_i, y_i);
                    int intensity = Color.red(pixel);
                    histogram[intensity]++;
                }
            }

            // Compute the cumulative distribution function (CDF) of the histogram
            int[] cdf = new int[256];
            cdf[0] = histogram[0];
            for (int j = 1; j < 256; j++) {
                cdf[j] = cdf[j - 1] + histogram[j];
            }

            // Compute the mapping function for each pixel intensity value
            int numPixels = processed.getWidth() * processed.getHeight();
            int[] mapping = new int[256];
            for (int j = 0; j < 256; j++) {
                float ratio = (float) cdf[j] / numPixels;
                mapping[j] = (int) (ratio * 255);
            }

            // Apply the mapping function to the pixel intensities of the grayscale image
            //Bitmap equalizedBitmap = Bitmap.createBitmap(processed.getWidth(), processed.getHeight(), Bitmap.Config.ARGB_8888);
            //Canvas canvas2 = new Canvas(equalizedBitmap);
            //Paint paint2 = new Paint();
            for (int x_i = 0; x_i < processed.getWidth(); x_i++) {
                for (int y_i = 0; y_i < processed.getHeight(); y_i++) {
                    int pixel = processed.getPixel(x_i, y_i);
                    int intensity = Color.red(pixel);
                    int equalizedIntensity = mapping[intensity];
                    int equalizedPixel = Color.rgb(equalizedIntensity, equalizedIntensity, equalizedIntensity);
                    //canvas2.drawPoint(x_i, y_i, paint2);
                    processed.setPixel(x_i, y_i, equalizedPixel);
                }
            }
        }

        endTime = System.nanoTime();
        equaliseTime = endTime - startTime;
        Trace.endSection();

        //---------------------
        // Scale
        //---------------------

        Bitmap resized = Bitmap.createScaledBitmap(processed, 112, 112, true);


        //---------------------
        // Bilateral Filter
        //---------------------

        Trace.beginSection("Blur");

        startTime = System.nanoTime();

        if (options.bilateral & options.useOpenCV){
            Utils.bitmapToMat(resized, mat);
            Mat filtered = new Mat(112, 112, CV_8UC3);

            cvtColor(mat, mat, CV_RGBA2RGB);

            Imgproc.bilateralFilter(mat, filtered , 3, 100, 100);
            Utils.matToBitmap(filtered, resized);
        }

        if (options.bilateral & !options.useOpenCV){
            RenderScript rs = RenderScript.create(mContext);

            Allocation inputAllocation = Allocation.createFromBitmap(rs, resized);
            Allocation outputAllocation = Allocation.createFromBitmap(rs, resized);

            ScriptIntrinsicBlur bilateralFilter = ScriptIntrinsicBlur.create(rs, Element.U8_4(rs));

            float radius = 1;

            bilateralFilter.setRadius(radius);

            bilateralFilter.setInput(inputAllocation);
            bilateralFilter.forEach(outputAllocation);

            outputAllocation.copyTo(resized);

            inputAllocation.destroy();
            outputAllocation.destroy();
            rs.destroy();
        }

        endTime = System.nanoTime();
        bilateralTime = endTime - startTime;
        Trace.endSection();
        Trace.endSection();
        return resized;
    }

    public long getAlignTime() {return alignTime;}
    public long getEqualiseTime() {return equaliseTime;}
    public long getGrayTime() {return grayTime;}
    public long getBilateralTime() {return bilateralTime;}


    public static class Options{
        boolean useOpenCV = false;
        boolean grayscale = false;
        boolean equalise = false;
        boolean rotate = false;
        boolean bilateral = false;

        public void setUseOpenCV(boolean useOpenCV){
            this.useOpenCV = useOpenCV;
        }

        public void setGrayscale(boolean grayscale){
            this.grayscale = grayscale;
        }

        public void setEqualise(boolean equalise){
            this.equalise = equalise;
        }

        public void setRotate(boolean rotate){
            this.rotate = rotate;
        }
        public void setBilateral(boolean bilateral){
            this.bilateral = bilateral;
        }
    }

}

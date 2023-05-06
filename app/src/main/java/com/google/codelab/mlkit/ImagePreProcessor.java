package com.google.codelab.mlkit;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.Log;

import com.google.mlkit.vision.face.Face;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class ImagePreProcessor {

    private Options options = new Options();

    public ImagePreProcessor(){}

    public ImagePreProcessor(Options options){
        this.options = options;
    }

    public Bitmap Process(Face face, Bitmap input){

        String landm = String.valueOf(face.getLandmark(0));
        Log.d("NSP debug: ", landm);

        int width = face.getBoundingBox().width();
        int height = face.getBoundingBox().height();
        float x = face.getBoundingBox().centerX();
        float y = face.getBoundingBox().centerY();
        float xOffset = width / 2.0f;
        float yOffset = height / 2.0f;
        float left = x - xOffset;
        float top = y - yOffset;

        Bitmap processed = Bitmap.createBitmap(input);

        Mat mat = new Mat();

        if (options.grayscale & options.useOpenCV & !options.equalise) {
            Utils.bitmapToMat(processed, mat);

            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGB2GRAY);

            Utils.matToBitmap(mat, processed);
        }

        if (options.grayscale & !options.useOpenCV) {
            Bitmap bm_grayscale = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);

            Canvas canvas = new Canvas(bm_grayscale);
            Paint paint = new Paint();
            ColorMatrix colorMatrix = new ColorMatrix();
            colorMatrix.setSaturation(0); // Set saturation to 0 to convert to grayscale
            paint.setColorFilter(new ColorMatrixColorFilter(colorMatrix));

            canvas.drawBitmap(processed, 0, 0, paint);

            processed = bm_grayscale;
        }

        if (options.rotate){

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
        processed = Bitmap.createBitmap(processed, (int) left, (int) top, width, height);

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


        Bitmap resized = Bitmap.createScaledBitmap(processed, 112, 112, false);

        return resized;
    }

    public static class Options{
        boolean useOpenCV = false;
        boolean grayscale = false;
        boolean equalise = false;
        boolean rotate = false;

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
    }

}

package com.capstone.objdetection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import com.capstone.objdetection.env.iBorderedText;
import com.capstone.objdetection.env.iImageUtils;
import com.capstone.objdetection.env.iLogger;
import com.capstone.objdetection.tracking.iMultiBoxTracker;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;


public class DetectorActivityI extends iCameraActivity implements OnImageAvailableListener {
  private static final iLogger I_LOGGER = new iLogger();

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final String TF_OD_API_MODEL_FILE =
      "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private iClassifier detector;

  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private iMultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private iBorderedText iBorderedText;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    iBorderedText = new iBorderedText(textSizePx);
    iBorderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new iMultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector = iTensorFlowObjectDetectionAPIModel.create(
          getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      I_LOGGER.e("Exception initializing classifier!", e);
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "iClassifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    I_LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    I_LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        iImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, false);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new iOverlayView.DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
          }
        });
  }

  iOverlayView trackingOverlay;

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
        previewWidth,
        previewHeight,
        sensorOrientation);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    I_LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            I_LOGGER.i("Running detection on image " + currTimestamp);
            final List<iClassifier.Recognition> results = detector.recognizeImage(croppedBitmap);

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

              final List<iClassifier.Recognition> mappedRecognitions =
                    new LinkedList<>();

            for (final iClassifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                I_LOGGER.i("Title: " + result.getTitle());
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            tracker.trackResults(mappedRecognitions);
            toSpeech(mappedRecognitions);
            trackingOverlay.postInvalidate();

            computingDetection = false;
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

}

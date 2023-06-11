const { google } = require('googleapis');

google.auth.getApplicationDefault(function (err, authClient, projectId) {
  if (err) {
    throw err;
  }

  if (authClient.createScopedRequired && authClient.createScopedRequired()) {
    authClient = authClient.createScoped([
      'https://www.googleapis.com/auth/cloud-platform',
      'https://www.googleapis.com/auth/userinfo.email'
    ]);
  }
  ml = google.ml('v1');
  ml.projects.jobs.create({
    auth: authClient,
    parent: 'projects/wizydam-dev',
    resource: {
      jobId: 'test_api_3',
      trainingInput: {
        "scaleTier": "CUSTOM",
        "masterType": "large_model",
        "args": [
          "--training_data_path=gs://wd-model-rnd-0000/cloud/test_bottles2/records/train-*-of-*.tfrecords",
          "--validation_data_path=gs://wd-model-rnd-0000/cloud/test_bottles2/records/validation-*-of-*.tfrecords",
          "--num_classes=11",
          "--max_steps=500",
          "--train_batch_size=12",
          "--num_eval_images=1",
          "--warmup_learning_rate=0.0001",
          "--initial_learning_rate=0.0001",
          "--learning_rate_decay_type=cosine",
          "--optimizer_type=momentum",
          "--optimizer_arguments=momentum=0.9",
          "--fpn_type=fpn",
          "--resnet_depth=50",
          "--max_num_bboxes_in_training=50",
          "--max_num_bboxes_in_prediction=50",
          "--anchor_size=4",
          "--image_size=640,640",
          "--bbox_aspect_ratios=1.0,2.0,0.5",
          "--fpn_min_level=3",
          "--fpn_max_level=7",
          "--nms_iou_threshold=0.5",
          "--nms_score_threshold=0.05",
          "--focal_loss_alpha=0.25",
          "--detection_loss_weight=50",
          "--auto_augmentation=false",
          "--aug_rand_hflip=false",
          "--aug_scale_min=1",
          "--aug_scale_max=1"
        ],
        "region": "us-central1",
        "jobDir": "gs://wd-model-rnd-0000/cloud/test_bottles2/model4",
        "masterConfig": {
          "imageUri": "gcr.io/cloud-ml-algos/image_object_detection:latest"
        },
      },
    }
  });
});


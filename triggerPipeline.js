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

  const dataflow = google.dataflow({ version: 'v1b3', auth: authClient });

  dataflow.projects.templates.create({
    projectId: projectId,
    resource: {
      parameters: {
        bq_dataset: `autolearningRND.bottlestTest2`,
        output_dir: 'gs://wd-model-rnd-0000/cloud/test_bottles/records',
        temp_dir: 'gs://wd-model-rnd-0000/cloud/test_bottles/tmp',
      },
      jobName: 'test-cloud-api-2',
      gcsPath: 'gs://wd-model-rnd-0000/dataflow_templates/MakeDatasetTemplate'
    }
  }, function(err, response) {
    if (err) {
      console.error("problem running dataflow template, error was: ", err);
    }
    console.log("Dataflow template response: ", response);
  });

});

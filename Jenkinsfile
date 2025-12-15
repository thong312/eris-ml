pipeline {
  agent any

  environment {
    DOCKER_IMAGE = "thong312/iris-ml-api"
    DOCKER_TAG = "${BUILD_NUMBER}"
    DOCKER_CREDENTIALS_ID = "dockerhub-credentials"
  }

  stages {

    stage('Checkout') {
      steps {
        echo 'Checking out code from Github ...'
        checkout scm
      }
    }

    stage('Setup Python Environment') {
      steps {
        echo 'Setting up Python virtual environment...'
        sh '''
          python3 -m venv venv
          . venv/bin/activate
          pip install --upgrade pip
          pip install -r requirements.txt
        '''
      }
    }

    stage('Train Model') {
      steps {
        echo 'Training ML model...'
        sh '''
          . venv/bin/activate
          cd src
          python train_model.py
          cd ..
        '''
      }
    }

    stage('Test Model') {
      steps {
        echo 'Testing model training and prediction...'
        sh '''
          . venv/bin/activate
          pytest tests/test_model.py -v --tb=short
        '''
      }
    }

    stage('Test API') {
      steps {
        echo 'Testing FastAPI application...'
        sh '''
          . venv/bin/activate
          pytest tests/test_app.py -v --tb=short
        '''
      }
    }

    stage('Build Docker image') {
      steps {
        echo 'Building Docker image...'
        script {
          docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
          docker.build("${DOCKER_IMAGE}:latest")
        }
      }
    }

    stage('Push to Docker Hub') {
      steps {
        echo 'Pushing Docker image to Docker Hub...'
        script {
          docker.withRegistry('https://index.docker.io/v1/', DOCKER_CREDENTIALS_ID) {
            docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push()
            docker.image("${DOCKER_IMAGE}:latest").push()
          }
        }
      }
    }

    stage('Cleanup') {
      steps {
        echo 'Cleaning up workspace...'
        sh '''
          docker rmi ${DOCKER_IMAGE}:${DOCKER_TAG} || true
          docker rmi ${DOCKER_IMAGE}:latest || true
          rm -rf venv
        '''
      }
    }
  }

  post {
    success {
      echo "Pipeline completed successfully!"
      echo "Docker image pushed: ${DOCKER_IMAGE}:${DOCKER_TAG}"
    }
    failure {
      echo "Pipeline failed!"
    }
    always {
      echo "Cleaning workspace"
      cleanWs()
    }
  }
}

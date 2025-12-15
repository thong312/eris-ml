pipeline {
  agent any

  environment {
    DOCKER_IMAGE = "username/iris-ml-api"
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

    stage('Setup Python Environment'){
        steps {
            echo 'Setting up Python virtual environment...'
            sh '''
            python3 -m venv venv
            source venv/bin/activate
            pip install --upgrade pip
            pip install -r requirements.txt
            '''
        }
    }
    stage('Train Model') {
      steps {
        echo 'Training ML model...'
        sh '''
        source venv/bin/activate
        cd src 
        python train_model.py
        cd ..
        '''
      }
    }

    stage('Test model') {
      steps { 
        echo "testing model training and prediction..."
        sh '''
        source venv/bin/activate
        pytest tests/test_model.py -v --tb=short
        '''
      }
    }
    stage('Test API'){
        echo 'Tesing FastAPI application...'
        sh '''
        source venv/bin/activate
        pytest tests/test_api.py -v --tb=short
        '''
    }
    stage('Build Docker image') {
      steps {
        echo 'Building Docker image...'
        sh '''
        docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}") .
        docker.build("${DOCKER_IMAGE}:latest") 
        '''
      }
    }

    stage('Push to Docker Hub') {
      steps {
        echo 'Pushing Docker image to Docker Hub...'
        scripts {
            docker.withRegistry('https://registry.hub.docker.com', "${DOCKER_CREDENTIALS_ID}") {
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
        echo 'Pipeline completed successfully!'
        echo Docker image pushed: ${DOCKER_IMAGE}:${DOCKER_TAG}
    }
    failure {
        echo 'Pipeline failed!'
    }
    always{
        echo "cleaning workspace"
        cleanWs()
    }
}
#№1 Download - скачивание датасета
sudo apt-get update # Обновим список пакетов
sudo apt-get install -y python3.12-venv # Установим venv для Python 3.12
python3 -m venv ./my_env # Создадим виртуальное окружение
. ./my_env/bin/activate # Активация окружения
python3 -m ensurepip --upgrade # Обновим pip
pip3 install setuptools # Установим setuptools
pip3 install -r requirements.txt # Установим зависимости
python3 download.py # Запустим скрипт загрузки

#№2 train_model - тренировка модели
. /var/lib/jenkins/workspace/download/my_env/bin/activate
cd /var/lib/jenkins/workspace/download # Перейдем в рабочую директорию
python3 train_model.py > best_model.txt # Обучение модели и сохранение вывода

#№3 deploy
. /var/lib/jenkins/workspace/download/my_env/bin/activate
cd /var/lib/jenkins/workspace/download 
export BUILD_ID=dontKillMe # Запретим Jenkins убивать процесс
path_model=$(cat best_model.txt | grep -oP '(?<=Путь к лучшей модели: ).*') # Извлечем путь к модели
mlflow models serve -m "$path_model" -p 5003 --no-conda & # Запустим сервис MLflow на порту 5003

#№4 healthy
. /var/lib/jenkins/workspace/download/my_env/bin/activate 
cd /var/lib/jenkins/workspace/download 
curl http://127.0.0.1:5003/invocations -H"Content-Type:application/json" --data '{"inputs": [[-1.275938045, -1.2340347, -1.41327673, 0.76150439, 2.20097247, -0.410937195, 0.58931542, 0.1135538, 0.58931542]]}' 
# Отправить тестовый запрос к модели

#Pipeline - для объедения задач в последовательный конвеер

pipeline {
    agent any

    stages {
        stage('Start Download') {
            steps {
                
                build job: "download"
                
            }
        }
        
        stage ('Train') {
            
            steps {
                
                script {
                    dir('/var/lib/jenkins/workspace/download') {
                        build job: "train_model"
                    }
                }
            
            }
        }
        
        stage ('Deploy') {
            steps {
                build job: 'deploy'
            }
        }
        
        stage ('Status') {
            steps {
                build job: 'healthy'
            }
        }
    }
}

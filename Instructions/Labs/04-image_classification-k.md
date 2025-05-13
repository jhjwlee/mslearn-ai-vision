# 이미지 분류 (Classify images)

Azure AI Custom Vision service를 사용하면 사용자의 자체 이미지로 학습된 computer vision model을 만들 수 있습니다. 이를 사용하여 이미지 분류(image classification) 및 객체 감지(object detection) model을 학습시킬 수 있으며, 그런 다음 애플리케이션에서 게시하고 사용할 수 있습니다.

*Note: Azure AI Custom Vision service - 사용자가 자체 이미지 데이터셋을 사용하여 특정 요구 사항에 맞는 computer vision model (이미지 분류, 객체 감지)을 학습, 배포 및 개선할 수 있도록 지원하는 서비스입니다.*
*Note: Image classification - 이미지가 어떤 사전 정의된 범주(class)에 속하는지 식별하는 작업입니다. 예를 들어, 이미지가 '사과', '바나나', '오렌지' 중 하나인지 분류합니다.*
*Note: Object detection - 이미지 내에서 특정 객체의 존재 여부와 위치(bounding box)를 식별하는 작업입니다.*

이 실습에서는 Custom Vision service를 사용하여 세 가지 종류의 과일(사과, 바나나, 오렌지)을 식별할 수 있는 이미지 분류 model을 학습합니다.

이 실습은 약 45분이 소요됩니다.

## Custom Vision resource 만들기 (Create Custom Vision resources)

Model을 학습하기 전에 학습(training) 및 예측(prediction)을 위한 Azure resource가 필요합니다. 이러한 각 작업에 대해 Custom Vision resource를 만들거나 단일 resource를 만들어 두 작업 모두에 사용할 수 있습니다. 이 실습에서는 학습 및 예측을 위한 Custom Vision resource를 만듭니다.

1.  Azure portal (https://portal.azure.com)을 열고 Azure 자격 증명을 사용하여 로그인합니다. 표시되는 모든 시작 메시지나 팁을 닫습니다.
2.  **Create a resource**를 선택합니다.
3.  검색 창에서 **Custom Vision**을 검색하고, **Custom Vision**을 선택한 후 다음 설정으로 resource를 만듭니다:
    *   **Create options**: Both
        *   *Note: 'Both' 옵션은 학습(training) resource와 예측(prediction) resource를 모두 생성합니다. 'Training only' 또는 'Prediction only'를 선택하여 특정 목적의 resource만 생성할 수도 있습니다.*
    *   **Subscription**: 사용자의 Azure subscription
    *   **Resource group**: resource group을 만들거나 선택합니다.
    *   **Region**: 사용 가능한 아무 region이나 선택합니다.
    *   **Name**: Custom Vision resource에 대한 유효한 이름
    *   **Training pricing tier**: F0
        *   *Note: F0는 무료 계층으로, 제한된 학습 시간과 이미지 저장 용량을 제공합니다.*
    *   **Prediction pricing tier**: F0
        *   *Note: F0는 무료 계층으로, 제한된 예측 트랜잭션 수를 제공합니다.*
4.  Resource를 만들고 배포가 완료될 때까지 기다린 다음 배포 세부 정보를 확인합니다. 두 개의 Custom Vision resource가 프로비저닝됩니다. 하나는 학습용이고 다른 하나는 예측용입니다.

*Note: 각 resource에는 자체 endpoint와 key가 있으며, 이는 코드에서 액세스를 관리하는 데 사용됩니다. 이미지 분류 model을 학습하려면 코드가 학습 resource(해당 endpoint 및 key 사용)를 사용해야 하며, 학습된 model을 사용하여 이미지 클래스를 예측하려면 코드가 예측 resource(해당 endpoint 및 key 사용)를 사용해야 합니다.*

Resource가 배포되면 resource group으로 이동하여 확인합니다. 접미사 `-Prediction`이 붙은 resource를 포함하여 두 개의 custom vision resource가 표시되어야 합니다.

## Custom Vision portal에서 Custom Vision project 만들기 (Create a Custom Vision project in the Custom Vision portal)

이미지 분류 model을 학습하려면 학습 resource를 기반으로 Custom Vision project를 만들어야 합니다. 이를 위해 Custom Vision portal을 사용합니다.

1.  새 브라우저 탭을 엽니다 (Azure portal 탭은 열어 두십시오 - 나중에 다시 돌아올 것입니다).
2.  새 브라우저 탭에서 Custom Vision portal (https://customvision.ai)을 엽니다. 메시지가 표시되면 Azure 자격 증명을 사용하여 로그인하고 서비스 약관에 동의합니다.
3.  Custom Vision portal에서 다음 설정으로 새 project를 만듭니다:
    *   **Name**: Classify Fruit
    *   **Description**: Image classification for fruit
    *   **Resource**: 사용자의 Custom Vision resource (학습용 resource를 선택합니다. 예측용 resource는 여기에 표시되지 않을 수 있습니다.)
    *   **Project Types**: Classification
        *   *Note: Project Types - 만들려는 model의 유형을 지정합니다. 'Classification' 또는 'Object Detection'을 선택할 수 있습니다.*
    *   **Classification Types**: Multiclass (single tag per image)
        *   *Note: Classification Types - 분류 유형을 지정합니다. 'Multiclass'는 각 이미지가 하나의 태그(범주)만 가질 수 있음을 의미합니다. 'Multilabel'은 각 이미지가 여러 태그를 가질 수 있음을 의미합니다.*
    *   **Domains**: Food
        *   *Note: Domains - project의 이미지 유형에 따라 최적화된 사전 학습된 model을 선택합니다. 'Food', 'Retail', 'Landmarks' 등 다양한 도메인이 있습니다. 'General' 도메인은 특정 최적화 없이 다양한 이미지에 사용됩니다.*
        *   *`Food` domain - 음식 이미지 분류에 최적화된 model입니다.*

## 이미지 업로드 및 태그 지정 (Upload and tag images)

1.  새 브라우저 탭에서 https://github.com/MicrosoftLearning/mslearn-ai-vision/raw/main/Labfiles/image-classification/training-images.zip 에서 학습 이미지를 다운로드하고 zip 폴더의 압축을 풀어 내용을 확인합니다. 이 폴더에는 사과(apple), 바나나(banana), 오렌지(orange) 이미지의 하위 폴더가 포함되어 있습니다.
2.  Custom Vision portal의 이미지 분류 project에서 **Add images**를 클릭하고, 이전에 다운로드하여 압축을 푼 `training-images/apple` 폴더의 모든 파일을 선택합니다. 그런 다음 다음과 같이 `apple` 태그를 지정하여 이미지 파일을 업로드합니다:

    (사과 태그로 사과 이미지 업로드 스크린샷)

3.  **Add Images** (**[+]**) 도구 모음 아이콘을 사용하여 이전 단계를 반복하여 `banana` 폴더의 이미지를 `banana` 태그로 업로드하고, `orange` 폴더의 이미지를 `orange` 태그로 업로드합니다.
4.  Custom Vision project에 업로드한 이미지를 탐색합니다. 각 클래스당 15개의 이미지가 있어야 합니다. 다음과 같습니다:

    (과일 태그가 지정된 이미지 - 사과 15개, 바나나 15개, 오렌지 15개)

## Model 학습 (Train a model)

1.  Custom Vision project에서 이미지 위쪽에 있는 **Train** (⚙⚙)을 클릭하여 태그가 지정된 이미지를 사용하여 분류 model을 학습합니다. **Quick Training** 옵션을 선택한 다음 학습 iteration이 완료될 때까지 기다립니다 (1분 정도 걸릴 수 있습니다).
    *   *Note: Quick Training - 상대적으로 적은 양의 데이터로 빠르게 model을 학습시키는 옵션입니다. 더 많은 시간과 데이터가 있다면 'Advanced Training'을 통해 더 나은 성능을 얻을 수 있습니다.*
    *   *Note: Iteration - model 학습의 한 주기입니다. 각 iteration은 이전 iteration의 결과를 바탕으로 model을 개선하려고 시도합니다.*
2.  Model iteration이 학습되면 **Precision**, **Recall**, **AP** 성능 지표를 검토합니다. 이 지표는 분류 model의 예측 정확도를 측정하며 모두 높아야 합니다.

    (Model 지표 스크린샷)

    *Note: Performance metrics - model의 성능을 평가하는 지표입니다.*
    *   *`Precision` (정밀도) - model이 특정 클래스로 예측한 항목 중 실제로 해당 클래스인 항목의 비율입니다. (TP / (TP + FP))*
    *   *`Recall` (재현율) - 실제 특정 클래스인 항목 중 model이 해당 클래스로 올바르게 예측한 항목의 비율입니다. (TP / (TP + FN))*
    *   *`AP` (Average Precision) - Precision-Recall 곡선 아래의 면적으로, model의 전반적인 성능을 나타내는 단일 지표입니다. 각 클래스별로 계산될 수 있으며, mAP(mean Average Precision)는 모든 클래스에 대한 AP의 평균입니다.*

    *Note: 성능 지표는 각 예측에 대해 50%의 확률 임계값(probability threshold)을 기준으로 합니다 (즉, model이 이미지가 특정 클래스일 확률을 50% 이상으로 계산하면 해당 클래스로 예측됨). 페이지 왼쪽 상단에서 이를 조정할 수 있습니다.*
    *   *Note: Probability threshold - model이 예측을 특정 클래스로 분류하기 위해 충족해야 하는 최소 확률 값입니다. 이 값을 조정하여 정밀도와 재현율 간의 균형을 맞출 수 있습니다.*

## Model 테스트 (Test the model)

1.  성능 지표 위에서 **Quick Test**를 클릭합니다.
2.  **Image URL** 상자에 `https://aka.ms/test-apple`을 입력하고 빠른 테스트 이미지 (➔) 버튼을 클릭합니다.
3.  Model이 반환한 예측을 확인합니다. `apple`에 대한 확률 점수가 가장 높아야 합니다. 다음과 같습니다:

    (사과 클래스 예측이 있는 이미지 스크린샷)

4.  다음 이미지를 테스트해 보십시오:
    *   `https://aka.ms/test-banana`
    *   `https://aka.ms/test-orange`
5.  **Quick Test** 창을 닫습니다.

## Project 설정 보기 (View the project settings)

만든 project에는 고유 식별자가 할당되었으며, 이 식별자는 project와 상호 작용하는 모든 코드에 지정해야 합니다.

1.  **Performance** 페이지 오른쪽 상단의 설정 (⚙) 아이콘을 클릭하여 project 설정을 봅니다.
2.  왼쪽의 **General** 아래에서 이 project를 고유하게 식별하는 **Project Id**를 확인합니다.
3.  오른쪽의 **Resources** 아래에 key와 endpoint가 표시되는지 확인합니다. 이는 학습 resource에 대한 세부 정보입니다 (Azure portal에서 resource를 확인하여 이 정보를 얻을 수도 있습니다).

## 학습 API 사용 (Use the training API)

Custom Vision portal은 이미지를 업로드하고 태그를 지정하며 model을 학습하는 데 사용할 수 있는 편리한 사용자 인터페이스를 제공합니다. 그러나 일부 시나리오에서는 Custom Vision training API를 사용하여 model 학습을 자동화할 수 있습니다.

*Note: 이 실습에서는 Python SDK에서 API를 사용합니다.*

### 애플리케이션 구성 준비 (Prepare the application configuration)

1.  Azure portal이 포함된 브라우저 탭으로 돌아갑니다 (Custom Vision portal 탭은 열어 두십시오 - 나중에 다시 돌아올 것입니다).
2.  Azure portal에서 페이지 상단의 검색 창 오른쪽에 있는 **\[>_]** 버튼을 사용하여 새 Cloud Shell을 만들고, 구독에 저장소 없이 PowerShell 환경을 선택합니다.

    Cloud Shell은 Azure portal 하단 창에 command-line interface를 제공합니다.

    *Note: 이전에 Bash 환경을 사용하는 Cloud Shell을 만든 경우 PowerShell로 전환하십시오.*

3.  Cloud Shell 도구 모음의 **Settings** 메뉴에서 **Go to Classic version**을 선택합니다 (코드 편집기를 사용하려면 이 작업이 필요합니다).

    계속하기 전에 Cloud Shell의 클래식 버전으로 전환했는지 확인하십시오.

4.  Cloud Shell 창의 크기를 조정하여 더 많이 볼 수 있도록 합니다.

    팁: 창 위쪽 테두리를 드래그하여 크기를 조절할 수 있습니다. 최소화 및 최대화 버튼을 사용하여 Cloud Shell과 기본 포털 인터페이스 간에 전환할 수도 있습니다.

5.  Cloud Shell 창에 다음 명령을 입력하여 이 실습용 코드 파일이 포함된 GitHub repo를 복제합니다 (명령을 입력하거나 클립보드에 복사한 다음 명령줄에서 마우스 오른쪽 버튼을 클릭하고 일반 텍스트로 붙여넣기):

    ```code
    rm -r mslearn-ai-vision -f
    git clone https://github.com/MicrosoftLearning/mslearn-ai-vision
    ```

    팁: Cloud Shell에 명령을 붙여넣을 때 출력이 화면 버퍼의 많은 부분을 차지할 수 있습니다. `cls` 명령을 입력하여 화면을 지우면 각 작업에 더 쉽게 집중할 수 있습니다.

6.  Repo가 복제된 후 다음 명령을 사용하여 애플리케이션 코드 파일이 포함된 Python용 폴더로 이동하여 파일을 확인합니다:

    **Python**
    ```code
    cd mslearn-ai-vision/Labfiles/image-classification/python/train-classifier
    ls -a -l
    ```
    이 폴더에는 앱의 애플리케이션 구성 및 코드 파일이 포함되어 있습니다. 또한 model의 추가 학습을 수행하는 데 사용할 일부 이미지 파일이 포함된 `/more-training-images` 하위 폴더도 포함되어 있습니다.

7.  다음 명령을 실행하여 학습용 Azure AI Custom Vision SDK 패키지 및 기타 필요한 패키지를 설치합니다:

    **Python**
    ```code
    python -m venv labenv
    ./labenv/bin/Activate.ps1
    pip install -r requirements.txt azure-cognitiveservices-vision-customvision
    ```
    *Note: `azure-cognitiveservices-vision-customvision` - Azure Custom Vision service와 상호 작용하기 위한 Python SDK입니다. 이 패키지에는 학습 API와 예측 API 모두에 대한 client가 포함되어 있습니다.*

8.  다음 명령을 입력하여 앱의 구성 파일을 편집합니다:

    **Python**
    ```code
    code .env
    ```

    파일이 코드 편집기에서 열립니다.

9.  코드 파일에서 Custom Vision **training** resource에 대한 **Endpoint**와 인증 **Key**, 그리고 이전에 만든 custom vision project의 **Project ID**를 반영하도록 구성 값을 업데이트합니다.
10. 자리 표시자를 바꾼 후 코드 편집기 내에서 `CTRL+S` 명령을 사용하여 변경 사항을 저장한 다음 `CTRL+Q` 명령을 사용하여 Cloud Shell 명령줄을 열어 둔 채로 코드 편집기를 닫습니다.

### Model 학습 수행 코드 작성 (Write code to perform model training)

1.  Cloud Shell 명령줄에서 다음 명령을 입력하여 클라이언트 애플리케이션의 코드 파일을 엽니다:

    **Python**
    ```code
    code train-classifier.py
    ```

2.  코드 파일에서 다음 세부 정보를 확인합니다:
    *   Azure AI Custom Vision SDK용 네임스페이스가 가져와집니다.
        *   *Note (Python): `from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient` 및 `from msrest.authentication import ApiKeyCredentials` 등이 사용됩니다.*
    *   `Main` 함수는 구성 설정을 검색하고, key와 endpoint를 사용하여 인증된 `CustomVisionTrainingClient`를 만듭니다. 그런 다음 project ID와 함께 사용하여 project에 대한 `Project` 참조를 만듭니다.
        *   *Note: `CustomVisionTrainingClient(endpoint, ApiKeyCredentials(api_key=training_key))` - 학습 API와 상호 작용하기 위한 client 객체를 생성합니다.*
    *   `Upload_Images` 함수는 Custom Vision project에 정의된 태그를 검색한 다음 해당 이름의 폴더에서 이미지 파일을 project에 업로드하고 적절한 태그 ID를 할당합니다.
        *   *Note: `trainer.create_images_from_files(project_id, images=image_list)` 또는 유사한 함수를 사용하여 이미지를 업로드하고 태그를 지정합니다.*
    *   `Train_Model` 함수는 project에 대한 새 학습 iteration을 만들고 학습이 완료될 때까지 기다립니다.
        *   *Note: `trainer.train_project(project_id)` - project 학습을 시작합니다. `trainer.get_iteration(project_id, iteration.id)`를 사용하여 학습 상태를 모니터링할 수 있습니다.*

3.  코드 편집기를 닫고 (`CTRL+Q`) 다음 명령을 입력하여 프로그램을 실행합니다:

    **Python**
    ```code
    python train-classifier.py
    ```

4.  프로그램이 종료될 때까지 기다립니다. 그런 다음 Custom Vision portal이 포함된 브라우저 탭으로 돌아가서 project의 **Training Images** 페이지를 봅니다 (필요한 경우 브라우저 새로 고침).
5.  일부 새로운 태그가 지정된 이미지가 project에 추가되었는지 확인합니다. 그런 다음 **Performance** 페이지를 보고 새 iteration이 만들어졌는지 확인합니다.

## 클라이언트 애플리케이션에서 이미지 분류기 사용 (Use the image classifier in a client application)

이제 학습된 model을 게시하고 클라이언트 애플리케이션에서 사용할 준비가 되었습니다.

### 이미지 분류 model 게시 (Publish the image classification model)

1.  Custom Vision portal의 **Performance** 페이지에서 **🗸 Publish**를 클릭하여 다음 설정으로 학습된 model을 게시합니다:
    *   **Model name**: fruit-classifier
        *   *Note: Model name - 게시된 model을 식별하는 데 사용되는 이름입니다. 예측 API를 호출할 때 이 이름을 사용합니다.*
    *   **Prediction Resource**: 이전에 만든 예측 resource (이름이 "-Prediction"으로 끝나는 resource, 학습 resource가 아님).
2.  **Project Settings** 페이지 왼쪽 상단에서 **Projects Gallery** (👁) 아이콘을 클릭하여 Custom Vision portal 홈 페이지로 돌아가면 project가 이제 나열됩니다.
3.  Custom Vision portal 홈 페이지 오른쪽 상단에서 설정 (⚙) 아이콘을 클릭하여 Custom Vision service의 설정을 봅니다. 그런 다음 **Resources** 아래에서 이름이 "-Prediction"으로 끝나는 예측 resource (학습 resource가 아님)를 찾아 해당 **Key**와 **Endpoint** 값을 확인합니다 (Azure portal에서 resource를 확인하여 이 정보를 얻을 수도 있습니다).

### 클라이언트 애플리케이션에서 이미지 분류기 사용 (Use the image classifier from a client application)

1.  Azure portal과 Cloud Shell 창이 포함된 브라우저 탭으로 돌아갑니다.
2.  Cloud Shell에서 다음 명령을 실행하여 클라이언트 애플리케이션용 폴더로 전환하고 포함된 파일을 봅니다:

    ```code
    cd ../test-classifier
    ls -a -l
    ```
    이 폴더에는 앱의 애플리케이션 구성 및 코드 파일이 포함되어 있습니다. 또한 model을 테스트하는 데 사용할 일부 이미지 파일이 포함된 `/test-images` 하위 폴더도 포함되어 있습니다.

3.  다음 명령을 실행하여 예측용 Azure AI Custom Vision SDK 패키지 및 기타 필요한 패키지를 설치합니다:

    **Python**
    ```code
    python -m venv labenv
    ./labenv/bin/Activate.ps1
    pip install -r requirements.txt azure-cognitiveservices-vision-customvision
    ```

4.  다음 명령을 입력하여 앱의 구성 파일을 편집합니다:

    **Python**
    ```code
    code .env
    ```

    파일이 코드 편집기에서 열립니다.

5.  Custom Vision **prediction** resource에 대한 **Endpoint**와 **Key**, 분류 project의 **Project ID**, 게시된 model의 이름 (이름은 `fruit-classifier`여야 함)을 반영하도록 구성 값을 업데이트합니다. 변경 사항을 저장 (`CTRL+S`) 하고 코드 편집기를 닫습니다 (`CTRL+Q`).
6.  Cloud Shell 명령줄에서 다음 명령을 입력하여 클라이언트 애플리케이션의 코드 파일을 엽니다:

    **Python**
    ```code
    code test-classifier.py
    ```

7.  코드를 검토하고 다음 세부 정보를 확인합니다:
    *   Azure AI Custom Vision SDK용 네임스페이스가 가져와집니다.
        *   *Note (Python): `from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient` 등이 사용됩니다.*
    *   `Main` 함수는 구성 설정을 검색하고, key와 endpoint를 사용하여 인증된 `CustomVisionPredictionClient`를 만듭니다.
        *   *Note: `CustomVisionPredictionClient(endpoint, ApiKeyCredentials(api_key=prediction_key))` - 예측 API와 상호 작용하기 위한 client 객체를 생성합니다.*
    *   예측 client 객체는 `test-images` 폴더의 각 이미지에 대한 클래스를 예측하는 데 사용되며, 각 요청에 대해 project ID와 model 이름을 지정합니다. 각 예측에는 가능한 각 클래스에 대한 확률이 포함되며, 확률이 50%보다 큰 예측된 태그만 표시됩니다.
        *   *Note: `predictor.classify_image(project_id, published_model_name, image_data)` 또는 `predictor.classify_image_url(project_id, published_model_name, url=image_url)`과 같은 함수를 사용하여 이미지를 분류합니다.*

8.  코드 편집기를 닫고 다음 SDK별 명령을 입력하여 프로그램을 실행합니다:

    **Python**
    ```code
    python test-classifier.py
    ```

9.  프로그램은 다음 각 이미지를 분류를 위해 model에 제출합니다:

    (사과 이미지)
    `IMG_TEST_1.jpg`

    (바나나 이미지)
    `IMG_TEST_2.jpg`

    (오렌지 이미지)
    `IMG_TEST_3.jpg`

    각 예측에 대한 레이블(태그)과 확률 점수를 확인합니다.

## 리소스 정리 (Clean up resources)

Azure AI Custom Vision 탐색을 마쳤으면 불필요한 Azure 비용이 발생하지 않도록 이 실습에서 만든 resource를 삭제해야 합니다:

1.  Azure portal (https://portal.azure.com)을 열고 상단 검색 창에서 이 실습에서 만든 resource를 검색합니다.
2.  Resource 페이지에서 **Delete**를 선택하고 지침에 따라 resource를 삭제합니다. 또는 전체 resource group을 삭제하여 모든 resource를 한 번에 정리할 수 있습니다.

---

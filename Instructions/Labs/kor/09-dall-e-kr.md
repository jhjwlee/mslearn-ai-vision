
## AI로 이미지 생성 (로컬 VS Code 환경)

이 실습에서는 OpenAI DALL-E 생성형 AI 모델을 사용하여 이미지를 생성합니다. 로컬 컴퓨터의 VS Code 환경에서 Azure AI Studio와 Azure OpenAI 서비스를 사용하여 앱을 개발합니다.

이 실습은 약 **30분**이 소요됩니다.

**학습 목표:**
이 실습을 통해 다음을 수행하는 방법을 배웁니다:
*   Azure AI Studio portal에 로그인하고 `dall-e-3` 모델을 선택하여 `project`를 생성합니다.
*   `Images playground`에서 배포된 DALL-E 모델을 테스트합니다.
*   로컬 VS Code 환경에서 Azure OpenAI SDK를 사용하여 DALL-E 모델과 상호작용하는 Python 클라이언트 애플리케이션을 개발합니다.
*   모델 엔드포인트에 연결하고, 텍스트 프롬프트를 사용하여 이미지 생성을 요청합니다.
*   생성된 이미지의 URL을 가져와 로컬에 파일로 저장합니다.

**주요 용어 (영문 유지):**
*   Azure AI Studio portal
*   Project
*   dall-e-3
*   Model deployment
*   Azure AI Studio resource (이전 Azure AI Foundry resource)
*   Subscription
*   Resource group
*   Region
*   Playgrounds
*   Images playground
*   Models + endpoints
*   Get endpoint
*   Local VS Code
*   Azure OpenAI SDK
*   `.env` file
*   `AIProjectClient`
*   `DefaultAzureCredential`
*   `get_azure_openai_client` (또는 유사한 메서드)
*   `images.generate()`

---

### Azure AI Studio portal 열기 (Open Azure AI Studio portal)

(이 부분은 환경과 무관하게 동일합니다.)

1.  웹 브라우저에서 Azure AI Studio portal (https://ai.azure.com)을 열고 Azure 자격 증명을 사용하여 로그인합니다. 처음 로그인할 때 열리는 모든 팁이나 빠른 시작 창을 닫고, 필요한 경우 왼쪽 상단의 Azure AI Studio 로고를 사용하여 홈페이지로 이동합니다.
2.  홈페이지의 정보를 검토합니다.

---

### 프로젝트를 시작할 모델 선택 (Choose a model to start a project)

(이 부분은 환경과 무관하게 동일합니다.)

1.  홈페이지의 **모델 및 기능 탐색(Explore models and capabilities)** 섹션에서 `project`에서 사용할 `dall-e-3` 모델을 검색합니다.
2.  검색 결과에서 `dall-e-3` 모델을 선택하여 세부 정보를 확인한 다음, 모델 페이지 상단에서 **이 모델 사용(Use this model)**을 선택합니다.
3.  `project`를 만들라는 메시지가 표시되면 `project`에 유효한 이름을 입력하고 **고급 옵션(Advanced options)**을 확장합니다.
4.  **사용자 지정(Customize)**을 선택하고 허브에 대해 다음 설정을 지정합니다:
    *   **Azure AI Studio resource:** `Azure AI Studio resource`에 대한 유효한 이름
    *   **Subscription:** 사용자의 Azure 구독
    *   **Resource group:** `Resource group` 만들기 또는 선택
    *   **Region:** AI 서비스가 지원되는 위치 선택\*
    \* 일부 Azure AI 리소스는 지역별 모델 할당량에 의해 제한됩니다. 실습 후반에 할당량 한도를 초과하는 경우 다른 지역에 다른 리소스를 만들어야 할 수 있습니다.
5.  **만들기(Create)**를 선택하고 선택한 `dall-e-3` 모델 배포를 포함하여 `project`가 생성될 때까지 기다립니다.
    **참고:** 모델 선택에 따라 `project` 생성 과정에서 추가 프롬프트가 표시될 수 있습니다. 약관에 동의하고 배포를 완료하십시오.
6.  `project`가 생성되면 **모델 + 엔드포인트(Models + endpoints)** 페이지에 모델이 표시됩니다.

---

### 플레이그라운드에서 모델 테스트 (Test the model in the playground)

(이 부분은 환경과 무관하게 동일합니다.)

1.  **플레이그라운드(Playgrounds)**를 선택한 다음, **이미지(Images) playground**를 선택합니다.
2.  DALL-E 모델 배포가 선택되어 있는지 확인합니다. 그런 다음 페이지 하단 근처의 상자에 "스파게티를 먹는 로봇 이미지 만들기(Create an image of an robot eating spaghetti)"와 같은 프롬프트를 입력하고 **생성(Generate)**을 선택합니다.
3.  `playground`에서 생성된 이미지를 검토합니다.
4.  "식당에 있는 로봇 보여주기(Show the robot in a restaurant)"와 같은 후속 프롬프트를 입력하고 생성된 이미지를 검토합니다.
5.  만족스러울 때까지 새 프롬프트로 계속 테스트하여 이미지를 구체화합니다.

---

### 클라이언트 애플리케이션 만들기 (Create a client application)

모델이 `playground`에서 잘 작동하는 것 같습니다. 이제 로컬 VS Code에서 Azure OpenAI SDK를 사용하여 클라이언트 애플리케이션에서 사용할 수 있습니다.

#### 애플리케이션 구성 준비 (Prepare the application configuration)

1.  Azure AI Studio portal에서 `project`의 **개요(Overview)** 페이지를 확인합니다.
2.  **모델 + 엔드포인트(Models + endpoints)** 영역을 선택한 다음, 배포한 DALL-E 모델을 선택하고 **엔드포인트 가져오기(Get endpoint)** 버튼 (또는 유사한 이름의 버튼)을 선택합니다. **엔드포인트(Endpoint)** 값과 **키(Key)** 값을 복사해 둡니다. (만약 키가 보이지 않고 엔드포인트 URI만 있다면 그것만 복사합니다. 인증은 `DefaultAzureCredential`을 사용할 것입니다.) 클라이언트 애플리케이션에서 모델에 연결하는 데 이 연결 문자열 또는 엔드포인트를 사용합니다.

3.  **로컬 개발 환경 설정:**
    a.  로컬 컴퓨터에 **Python 3.8 이상**이 설치되어 있는지 확인합니다.
    b.  로컬 컴퓨터에 **Git**이 설치되어 있는지 확인합니다.
    c.  로컬 컴퓨터에 **Visual Studio Code (VS Code)**가 설치되어 있는지 확인합니다.
    d.  VS Code에 **Python 확장 프로그램**이 설치되어 있는지 확인합니다.

4.  **프로젝트 폴더 생성 및 Git 리포지토리 클론:**
    a.  로컬 컴퓨터에서 이 실습을 위한 새 폴더를 만듭니다 (예: `dalle_project`).
    b.  VS Code를 열고, **파일(File) > 폴더 열기(Open Folder...)**를 선택하여 방금 만든 폴더를 엽니다.
    c.  VS Code에서 터미널을 엽니다 (**보기(View) > 터미널(Terminal)** 또는 `Ctrl+``).
    d.  터미널에서 다음 명령을 실행하여 실습 파일이 포함된 GitHub 리포지토리를 클론합니다:
        ```bash
        git clone https://github.com/MicrosoftLearning/mslearn-ai-vision
        ```
    e.  터미널에서 다음 명령을 실행하여 이 실습에 필요한 파일이 있는 폴더로 이동합니다:
        ```bash
        cd mslearn-ai-vision/Labfiles/dalle-client/python
        ```
    *   **핸즈온에서 배울 점:**
        *   로컬 개발 환경을 준비하고, Git을 사용하여 필요한 코드를 로컬 머신으로 가져오는 방법을 익힙니다.

5.  **Python 가상 환경 설정 및 라이브러리 설치:**
    a.  현재 위치 (`mslearn-ai-vision/Labfiles/dalle-client/python`)의 터미널에서 다음 명령을 실행하여 Python 가상 환경을 만듭니다:
        ```bash
        python -m venv .venv
        ```
    b.  가상 환경을 활성화합니다:
        *   Windows:
            ```bash
            .venv\Scripts\activate
            ```
        *   macOS/Linux:
            ```bash
            source .venv/bin/activate
            ```
        터미널 프롬프트 앞에 `(.venv)`가 표시되어야 합니다.
    c.  필요한 라이브러리를 설치합니다:
        ```bash
        pip install -r requirements.txt
        pip install azure-identity azure-ai-projects openai requests # requirements.txt에 이미 있거나 추가 설치
        ```
    *   **핸즈온에서 배울 점:**
        *   로컬 Python 프로젝트를 위한 가상 환경을 설정하고 활성화하는 방법을 배웁니다.
        *   `requirements.txt` 및 `pip install`을 사용하여 필요한 패키지를 로컬 가상 환경에 설치합니다.

6.  **구성 파일 설정 (`.env`):**
    a.  VS Code 탐색기에서 현재 폴더 (`mslearn-ai-vision/Labfiles/dalle-client/python/`)에 `.env`라는 새 파일을 만듭니다.
    b.  생성된 `.env` 파일에 다음 내용을 추가하고, 자리 표시자를 2단계에서 복사한 실제 값으로 바꿉니다:

        ```env
        AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
        AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
        AZURE_OPENAI_DEPLOYMENT_NAME="your_model_deployment_name"
        PROJECT_ENDPOINT="your_project_endpoint_from_ai_studio" # Azure AI Studio 프로젝트 엔드포인트
        ```
        *   `your_azure_openai_endpoint`: Azure AI Studio의 `Models + endpoints` 페이지에서 DALL-E 배포의 **엔드포인트(Endpoint)** 값을 여기에 입력합니다.
        *   `your_azure_openai_api_key`: 해당 엔드포인트의 **키(Key)** 값을 여기에 입력합니다.
        *   `your_model_deployment_name`: Azure AI Studio에서 DALL-E 모델 배포에 지정한 이름 (예: `dall-e-3`).
        *   `your_project_endpoint_from_ai_studio`: 이전 실습에서 사용한 것과 같은 Azure AI Studio `project`의 엔드포인트입니다. `AIProjectClient`를 초기화하는 데 사용됩니다.
    c.  파일을 저장합니다.
    *   **핸즈온에서 배울 점:**
        *   로컬 프로젝트에서 `.env` 파일을 사용하여 민감한 구성 정보를 안전하게 관리하는 방법을 익힙니다.

#### 프로젝트에 연결하고 모델과 이미지를 생성하는 코드 작성 (Write code to connect to your project and generate images with your model)

(Python 코드 자체는 이전 답변과 동일합니다. 파일 경로와 VS Code 사용법만 로컬 환경에 맞게 조정됩니다.)

1.  VS Code 탐색기에서 `dalle-client.py` 파일을 찾아 엽니다.
2.  `Add references` 주석 아래에 필요한 라이브러리를 가져오는 코드를 추가합니다 (이전 답변의 코드와 동일):

    ```python
    # Add references
    import os
    import json
    from dotenv import load_dotenv
    from azure.identity import DefaultAzureCredential
    from azure.ai.projects import AIProjectClient
    from openai import AzureOpenAI
    import requests
    ```

3.  `main` 함수에서 `Get configuration settings` 주석 아래 코드가 `.env` 파일 값을 로드하는지 확인합니다 (이전 답변의 코드와 동일).

4.  `Initialize the client` 주석 아래에 Azure OpenAI 클라이언트를 초기화하는 코드를 추가합니다. **실습 가이드 원본에서 제공된 Python 코드를 따르는 것이 가장 중요합니다.** 만약 가이드가 `AIProjectClient`를 통해 `get_azure_openai_client`를 사용하는 방식이라면 해당 코드를 사용합니다.

    ```python
    # Initialize the client
    # AIProjectClient를 사용하여 AI Studio 프로젝트에 연결
    project_client = AIProjectClient(
        endpoint=project_connection_string, # .env에서 로드한 AI Studio 프로젝트 엔드포인트
        credential=DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_managed_identity_credential=True
        )
    )

    # AI Studio 프로젝트를 통해 Azure OpenAI 클라이언트 가져오기
    # API 버전은 DALL-E 3를 지원하는 최신 안정 버전으로 조정해야 할 수 있습니다.
    # Azure OpenAI Studio의 모델 배포 세부 정보에서 권장 API 버전을 확인할 수 있습니다.
    openai_client = project_client.inference.get_azure_openai_client(api_version="2024-02-01") # 예시 API 버전
    ```
    *   **`DefaultAzureCredential` 로컬 환경:** 로컬 환경에서는 `DefaultAzureCredential`이 Azure CLI 로그인, VS Code Azure 계정 확장 프로그램 로그인, 환경 변수 등 다양한 방법을 통해 자격 증명을 찾으려고 시도합니다. 일반적으로 Azure CLI에 로그인되어 있으면 잘 작동합니다.

5.  루프 섹션의 `Generate an image` 주석 아래에 이미지를 생성하고 URL을 검색하는 코드를 추가합니다 (이전 답변의 코드와 동일):

    ```python
    # 루프 내:
    # Generate an image
    result = openai_client.images.generate(
        model=model_deployment,
        prompt=user_input,
        n=1,
        size="1024x1024"
    )

    json_response = json.loads(result.model_dump_json())
    image_url = json_response["data"][0]["url"]
    print(f"Image generated: {image_url}")
    ```

6.  이미지를 다운로드하는 `download_image` 함수가 있는지 확인하고, `main` 함수 내에서 호출하는 부분을 작성합니다 (이전 답변의 코드와 동일).

    ```python
    def download_image(url, filename):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Image saved as {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {e}")

    # ... main 함수 ...
    # main 함수 내 루프에서 image_url을 얻은 후
    if image_url:
        # images 디렉토리가 없다면 생성
        if not os.path.exists("images"):
            os.makedirs("images")
        
        # 간단한 파일명 생성 (예: image_1.png, image_2.png 등)
        # 이미 생성된 파일 수를 세어 다음 번호로 저장
        existing_files = len([name for name in os.listdir("images") if os.path.isfile(os.path.join("images", name)) and name.startswith("image_") and name.endswith(".png")])
        file_number = existing_files + 1
        filename = f"images/image_{file_number}.png"
        download_image(image_url, filename)
    ```

7.  VS Code에서 파일을 저장합니다 (Ctrl+S 또는 Cmd+S).

---

### 클라이언트 애플리케이션 실행 (Run the client application)

1.  **Azure CLI 로그인 (로컬 터미널):**
    VS Code 터미널 (가상 환경이 활성화된 상태)에서 Azure에 로그인되어 있는지 확인합니다. 로그인되어 있지 않다면 다음 명령을 실행합니다:
    ```bash
    az login
    ```
    브라우저가 열리고 로그인하라는 메시지가 표시됩니다. 로그인 프로세스를 완료합니다.
    *   **핸즈온에서 배울 점:**
        *   로컬 개발 환경에서 Azure CLI를 사용하여 Azure에 인증하는 방법을 익힙니다. `DefaultAzureCredential`이 이 로그인 정보를 사용합니다.

2.  **애플리케이션 실행:**
    VS Code 터미널에서 다음 명령을 입력하여 앱을 실행합니다:
    ```bash
    python dalle-client.py
    ```
3.  메시지가 표시되면 "피자 먹는 로봇 이미지 만들기(Create an image of a robot eating pizza)"와 같은 이미지 요청을 입력합니다. 잠시 후 앱에서 이미지가 저장되었음을 확인해야 합니다. 생성된 이미지는 프로젝트 폴더 내의 `images` 하위 폴더에 저장됩니다 (예: `dalle_project/mslearn-ai-vision/Labfiles/dalle-client/python/images/image_1.png`).
4.  몇 가지 프롬프트를 더 시도해 보십시오. 완료되면 "quit"을 입력하여 프로그램을 종료합니다.

5.  **생성된 이미지 확인:**
    VS Code 탐색기 또는 로컬 파일 탐색기를 사용하여 `images` 폴더로 이동하여 생성된 `.png` 파일을 열어 이미지를 확인합니다.
    *   **핸즈온에서 배울 점:**
        *   로컬 VS Code 환경에서 Python 스크립트를 실행하고, 생성된 결과물(이미지 파일)을 로컬 파일 시스템에서 직접 확인하는 방법을 배웁니다.

---

### 요약 (Summary)

이 실습에서는 로컬 VS Code 환경에서 Azure AI Studio와 Azure OpenAI SDK를 사용하여 DALL-E 모델을 사용하여 이미지를 생성하는 클라이언트 애플리케이션을 만들었습니다.

---

### 정리 (Clean up)

(이 부분은 환경과 무관하게 동일합니다.)

DALL-E 탐색을 마쳤다면 불필요한 Azure 비용이 발생하지 않도록 이 실습에서 만든 리소스를 삭제해야 합니다.

1.  Azure Portal이 포함된 브라우저 탭으로 돌아가거나 새 브라우저 탭에서 Azure Portal (https://portal.azure.com)을 다시 엽니다.
2.  이 실습에서 사용한 리소스를 배포한 `resource group`의 내용을 확인합니다.
3.  도구 모음에서 **리소스 그룹 삭제(Delete resource group)**를 선택합니다.
4.  `resource group` 이름을 입력하고 삭제할 것인지 확인합니다.

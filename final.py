
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from bokeh.plotting import figure


# Webpage Title
st.title("Word2vec 모델을 통한 기업의 조직 문화 측정")

st.write("김라윤(2022-33318), 신승연(2022-27512), 이선율(2021-28347), 장승현(2022-27575)")
st.markdown("***")

st.header("1. 연구 목적")
st.markdown(
    """
    _최근 빅 데이터(big-data) 분석에서 사용되고 있는 텍스트 마이닝(text-mining) 방법론은 발전된 정보처리 기술과 인프라를 활용하여 뉴스, 인터넷 등의 텍스트 문서로부터 정보를 획득하고, 키워드의 패턴을 분석하여 예측을 수행하는 방법론이다. 텍스트 마이닝은 데이터 마이닝과 유사한 개념이지만 기존 데이터 마이닝이 관계형 데이터베이스나 XML과 같은 구조화된 데이터들만을 처리하였다면, 텍스트 마이닝은 텍스트 문서, e-메일, HTML 파일과 같은 비정형 또는 반정형 데이터를 일정한 형식과 조건을 만족하는 자료로 가공하여 분석하며, 최근 연구 분야에서도 그 활용 영역을 점차 확장해 나가고 있다 (최정원, 한호선, 이미영, 안준모, 2015)._
    """
)


st.markdown(
    """
    * 텍스트 분석은 소비자 분석, 제품 분석, 고객관리 영역 등 다방변에서 활용
    * 인사-조직 분야에서 텍스트 분석은 그 용도가 매우 제한적으로 활용
    * 인사조직 분야는 전통적으로 설문, 질적 인터뷰, 기업의 패널 데이터를 사용하여 연구를 수행
    * 기업의 특성상 데이터 수집이 기업 활동에 방해가 됨으로 데이터 수집이 매우 어려움
    * 텍스트 분석은 인사-조직분야의 연구에 새로운 돌파구가 될 수 있음
    """
)

st.markdown(
    """
    * 본 연구에서는 온라인 기업 리뷰 사이트인 [잡플래닛](http://jobplanet.com)에 등록된 국내 제조/화학 산업군 기업의 리뷰 데이터를 활용해 새로운 조직문화 측정 방법론을 제시하고자 함
    * 또한, 새로운 방법으로 측정된 조직 문화 변수가 다른 기업 수준 변수인 연간 퇴사자율이나 구성원의 조직 만족도와 같은 변수들을 설명력 있게 예측할 수 있는지를 알아보기 위해 추가적인 분석을 수행하고, 변수 측정치의 타당성을 검증하는 것을 최종 목적 함
    """
)



st.info(
"""
1) Kai Li, Feng Mai, Rui Shen, Xinyan Yan (2020) Measuring Corporate Culture Using Machine Learning
- Word2vec 모델을 사용하여 기업 문화 단어 사전을 생성한 후 기업의 문화를 측정
- 2001-2018년 간 기업의 Earning calls data를 사용

2) 최진욱, 신동원, 이한준 (2021) IT 기업 직원의 만족 및 불만족 요인에 따른 이직률 예측: 토픽모델링과 머신러닝을 활용하여
- IT 기업 직원의 이직율 예측 모델을 구축
- 온라인 기업 리뷰 사이트인 잡플래닛에서 129개의 IT 기업에 종사했던 직원들 리뷰를 토픽 모델링을 시행하여 토픽 추출
- 추출된 토픽으로 머신러닝 기반 예측 모델 구축
"""
)

st.subheader("조직문화 유형: Cameron & Quinn (1980) Competeing Values Framework")
st.markdown(
    """
    * 조직 효과성(organizational effectiveness)의 지표 30개를 2개의 가치 차원(value dimensions)으로 분류
    * 2개의 가치 차원은 조직의 효과성과 관련됨
    * 조직 초점(Organizational Focus): 내부(통합) vs 외부(적응)
    * 조직 구조(Organizational Structure): 안정성(질서와 통제) vs 유연성(혁신과 변화)
    * 2개의 차원(X축,Y축)을 기준으로 기업의 문화를 4개의 유형으로 구분
    * 혁신(Adhocracy), 시장(Market), 집단(Clan), 위계(Hierachy)
    """
)

from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\사진파일\조직문화유형.png")
st.image(image)


st.subheader("문화 별 특징")
option = st.selectbox('문화 종류를 선택해주세요.',
                     ('시장 문화', '집단 문화', '위계 문화', '혁신 문화'))

if option == '시장 문화':
    st.write(option, ' : 외부지향적, 안정성')
    st.write("생산성/효율성, 성과지향적, 목표달성, 외부영입, 경직 구조, 매뉴얼/절차, 시장")

if option == '집단 문화':
    st.write(option, ' : 내부지향적, 유연성')
    st.write("내부육성/개발, 헌신 기반, 가부장적, 팀워크, 응집력, 자율권, 재량권")

if option == '위계 문화':
    st.write(option, ' : 내부지향적, 유연성')
    st.write("위계질서, 상명하달, 경직 구조, 통제, 권한 제한적, 내부육성/개발")

if option == '혁신문화':
    st.write(option, ' : 외부지향적, 유연성')
    st.write("창의성/혁신, 성과중심, 외부영입, 유연 구조, 재량권/자율권")

st.markdown("***")

st.header("2. 연구 절차")
st.subheader("1) Seed Words 설정")
st.write('Seed Words 는 Word2vec 모델 학습 시 학습의 기준이 되는 단어목록들')
st.write('기존 경영학 논문들에서 경쟁가치모형에 기반한 기업문화 4유형을 설명하는 핵심 단어들을 수집')
st.write('수집한 단어목록 중 유형 간 경계가 애매하거나, 범용적으로 사용되는 단어를 제거')
st.write('문화 유형을 가장 효과적으로 설명하면서 다른 유형과 배제되는 단어 4개를 선택')

st.info("""1) 혁신 문화: 창의, 도전, 자유, 변화
2) 시장 문화: 효율, 정형, 절차, 목표달성, 실적
3) 집단 문화: 가족, 육성, 배려, 충성심
4) 위계 문화: 권위, 명령, 보수, 질서""")

st.markdown("***")

st.subheader("2) 잡플래닛 데이터 크롤링")
st.info("https://drive.google.com/file/d/1fuiC80pDOaygsHy6_rcxhEjxLYMuNLIQ/view?usp=sharing")
st.markdown(
    """
    * 우리나라 최대 구인구직 사이트인 잡플래닛의 구직자 리뷰 텍스트를 크롤링
    * 제조/화학 산업군의 100개 기업을 대상으로 데이터를 수집
    * 가장 리뷰 수가 많은 10개의 기업을 대상으로 모든 리뷰 수집, 이후 90개 기업은 기업 당 450개의 리뷰 수집하여총 74,040개 리뷰 수집
    * 가장 리뷰 수가 많은 10개 기업 대상 기업문화를 측정 및 문화 점수 시각화
    """
)

from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\사진파일\9조 Wordvec 모델을 통한 기업문화 측정_1.jpg")
st.image(image)



cb1 = st.checkbox('모듈 및 라이브러리 임포트')	
if cb1:
    st.code(
        """
        pip install selenium
        # Pd
        import pandas as pd
        # BS
        from bs4 import BeautifulSoup
        
        # Se
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        import time
        
        # tqdm은 for 반복문이 얼만큼 진행했는지 알려주는 라이브러리
        from tqdm import tqdm
        """
        )

cb2 = st.checkbox('드라이버 설정 및 장착')
if cb2:
    st.code(
        """
        !apt-get update
        !apt install chromium-chromedriver
        !cp /usr/lib/chromium-browser/chromedriver /usr/bin

        from selenium import webdriver as wd

        options = wd.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = wd.Chrome('chromedriver', options=options)
        """
        )

cb3 = st.checkbox('(잡플래닛) 로그인 기능 구현')
if cb3:
    st.code(
        """
        url = "https://www.jobplanet.co.kr/users/sign_in?_nav=gb"
        driver.get(url)
        time.sleep(2)

        usr_id = "danielflash@snu.ac.kr"
        usr_pw = "snu960821@"


        # 1) e-mail
        driver.find_element(by = By.CSS_SELECTOR, value = "#user_email").send_keys(usr_id)

        # 2) Pw
        driver.find_element(by = By.CSS_SELECTOR, value = "#user_password").send_keys(usr_pw)

        # 3) '로그인' 버튼 클릭
        driver.find_element(by = By.CSS_SELECTOR, value = "#signInSignInCon > div.signInsignIn_wrap > div > section.section_email.er_r > fieldset > button").click()

        time.sleep(2)
        """
    )

cb4 = st.checkbox('(잡플래닛) 기업 정보 수집: 제조업 분야')
if cb4:
    st.code(
        """
        # 최초접속부분
        url = f"https://www.jobplanet.co.kr/companies?industry_id=200&page=1"
        driver.get(url)
        time.sleep(2)

        # 페이지 마지막 정보
        last_page_url = driver.find_element(by = By.CLASS_NAME, value= "btn_pglast").get_attribute("href")
        last_page_num = int(last_page_url.split("page=")[1])

        company_info_list = []

        ########### 페이지 순환 조절부분 ################
        for page in range(1,last_page_num+1)[:10] :
            url = f"https://www.jobplanet.co.kr/companies?industry_id=200&page={page}"
            driver.get(url)
            time.sleep(2)

            box_elements = driver.find_elements(by=By.CLASS_NAME, value = "content_wrap")

            for box_element in box_elements :
                company_id = box_element.find_element(by=By.CSS_SELECTOR, value="button").get_attribute("data-company_id")
                
                company_name = box_element.find_element(by=By.CSS_SELECTOR, value="#listCompanies > div > div.section_group > section > div > div > dl.content_col2_3.cominfo > dt > a").text
                # 수정
                company_url = box_element.find_elements(by=By.CLASS_NAME, value="us_stxt_1")[2].get_attribute("href")
                
                comapny_industry = box_element.find_elements(by=By.CLASS_NAME, value="us_stxt_1")[0].text
                comapny_region = box_element.find_elements(by=By.CLASS_NAME, value="us_stxt_1")[1].text
                comapny_review_count = box_element.find_elements(by=By.CLASS_NAME, value="us_stxt_1")[2].text.split()[0]
                comapny_salary_count = box_element.find_elements(by=By.CLASS_NAME, value="us_stxt_1")[3].text.split()[0]
                comapny_interview_count = box_element.find_elements(by=By.CLASS_NAME, value="us_stxt_1")[4].text.split()[0]
                comapny_salary_avg = box_element.find_elements(by=By.CLASS_NAME, value="us_stxt_1")[5].text.split()[1].replace(",","")
                
                company_info_list.append((company_id, company_name, company_url, comapny_industry, comapny_region, 
                                        comapny_review_count, comapny_salary_count, comapny_interview_count, comapny_salary_avg ))

                
        company_df = pd.DataFrame(company_info_list, columns=['company_id', 'company_name', 'company_url', 
                                                            'comapny_industry', 'comapny_region', 
                                                            'comapny_review_count', 'comapny_salary_count', 'comapny_interview_count', 
                                                            'comapny_salary_avg'])
        """
    )

cb5 = st.checkbox('(잡플래닛) 기업 리뷰 수집: 기업, 페이지 순환')
if cb4:
    st.code(
        """
        ##### 리뷰 저장  ##########
        review_info_list = []

        ############ 기업수 조절부분 ###########
        for company_url in company_df["company_url"][40:50] :
            
            driver.get(company_url)
            time.sleep(2)
            
            # 팝업 제거 용
            try : 
                driver.find_element(by = By.CSS_SELECTOR, value = "#premiumReviewChart > div > div.layer_popup_box.layer_popup_box_on > div.layer_popup.jply_modal.premium_review_inform > div > div.premium_modal_header > button").send_keys("\n")
            except :
                pass
            
            # 1. 페이지 순환용
            # 1-1. 마지막 페이지 URL 정보를 마지막 페이지 버튼에서 찾는다
            last_page_url = driver.find_element(by= By.CLASS_NAME, value= "btn_pglast").get_attribute("href")
            # 1-2. 마지막 페이지 URL를  Split하여 마지막 숫자값만 남김
            last_page_num = int(last_page_url.split("page=")[1])

            # 2. 기업 순환용
            # 2-1. 마지막 페이지 URL를 Split하여 URL값(https://www.jobplanet.co.kr/companies/30139/reviews/삼성전자?)만 남김
            page_url = last_page_url.split("page=")[0] 

            ############## 페이지 조절부분 #############
            for page in range(1, last_page_num+1)[:90] :
                url = f"{page_url}page={page}"
                
                driver.get(url)
                time.sleep(2)
                
                try:
                    # CLASS_NAME이 content_wrap인 element가 로딩될 때 까지 10초 대기
                    element = WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.CLASS_NAME, "content_wrap")))
                except TimeoutException:
                    # 실패 시에는 에러메시지로 Time Out 출력
                    print('Time Out')
                        
                box_elements = driver.find_elements(by= By.CLASS_NAME, value="content_wrap")

                for box_element in box_elements :

                    # usr
                    usr_info = box_element.find_element(by = By.CLASS_NAME, value= "content_top_ty2").text
                    try : 
                        usr_job = usr_info.split("|")[0].strip()
                    except :
                        usr_job = ""
                    try : 
                        usr_status = usr_info.split("|")[1].strip()
                    except :
                        usr_status = ""
                    try :
                        usr_region = usr_info.split("|")[2].replace("\n기업 추천 리뷰","").strip()
                    except :
                        usr_region = ""
                    try :
                        usr_date= usr_info.split("|")[3].strip()
                    except : 
                        usr_date= ""
                        
                    # score
                    try : 
                        star_score = box_element.find_element(by = By.CLASS_NAME, value="star_score").get_attribute("style").split(': ')[1].split('%')[0]
                    except : 
                        star_score = ""
                    try: 
                        promo_score = box_element.find_elements(by = By.CLASS_NAME, value="bl_score")[0].get_attribute("style").split(': ')[1].split('%')[0]
                    except :
                        promo_score = ""
                    try: 
                        salary_score = box_element.find_elements(by = By.CLASS_NAME, value="bl_score")[1].get_attribute("style").split(': ')[1].split('%')[0]
                    except :
                        salary_score=""
                    try: 
                        balance_score = box_element.find_elements(by = By.CLASS_NAME, value="bl_score")[2].get_attribute("style").split(': ')[1].split('%')[0]
                    except :
                        balance_score=""
                    try: 
                        culture_score = box_element.find_elements(by = By.CLASS_NAME, value="bl_score")[3].get_attribute("style").split(': ')[1].split('%')[0]
                    except :
                        culture_score = ""
                    try: 
                        manage_score = box_element.find_elements(by = By.CLASS_NAME, value="bl_score")[4].get_attribute("style").split(': ')[1].split('%')[0]         
                    except :
                        manage_score = ""
                                                                                                                                                            

                    # review 
                    try: 
                        review_summary = box_element.find_element(by = By.CLASS_NAME, value="us_label").text.replace('"',"")
                    except :
                        review_summary = ""
                    try: 
                        review_merit = box_element.find_element(by = By.CSS_SELECTOR, value="#viewReviewsList > div > div > div > section > div > div.ctbody_col2 > div > dl > dd:nth-child(2) > span").text
                    except :
                        review_merit =""
                    try: 
                        review_disadvantages = box_element.find_element(by = By.CSS_SELECTOR, value="#viewReviewsList > div > div > div > section > div > div.ctbody_col2 > div > dl > dd:nth-child(4) > span").text
                    except :
                        review_disadvantages=""
                    try: 
                        review_management = box_element.find_element(by = By.CSS_SELECTOR, value="#viewReviewsList > div > div > div > section > div > div.ctbody_col2 > div > dl > dd:nth-child(6) > span").text
                    except :
                        review_management=""
                    try: 
                        review_etc = box_element.find_element(by = By.CLASS_NAME, value="etc_box").text.replace("이 기업을 추천하지 않습니다.", "")
                    except :
                        review_etc=""
                    try:
                        review_recommend = box_element.find_element(by = By.CLASS_NAME, value="txt.recommend.etc_box").text
                    except : 
                        review_recommend=""

                    review_info_list.append((company_url,
                                            usr_job, usr_status, usr_region, usr_date,
                                            star_score, promo_score, salary_score, balance_score, culture_score, manage_score,
                                            review_summary, review_merit, review_disadvantages, review_management, 
                                            review_etc, review_recommend)) 

                    
        review_df = pd.DataFrame(review_info_list, columns = ['company_url',
                                                'usr_job', 'usr_status','usr_region','usr_date',
                                                'star_score', 'promo_score', 'salary_score', 'balance_score', 'culture_score', 'manage_score',
                                                'review_summary','review_merit', 'review_disadvantages','review_management', 'review_etc', 'review_recommend'])
        """
    )

cb6 = st.checkbox('기업정보 + 기업리뷰 Merge')
if cb6:
    st.code(
        """
        final_df = pd.merge(company_df[['company_name','company_url']],review_df, on = "company_url") 
        """
    )

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    df = pd.read_csv("C:\\Users\\82105\\Documents\\카카오톡 받은 파일\\final_df[0-100]85010.csv")
    st.write(df)

st.markdown("***")





st.subheader("3) 데이터 전처리")
st.info("https://drive.google.com/file/d/1N6LkncRhb7-gnGv4PznbkuBNlDNe2UqK/view?usp=sharing")
st.write('3-1 정규표현식 적용')
st.text('한글자 단어, 영어, 특수문자 제거')
st. code(
    """
    import re

    def apply_regular_expression(text):
    hangul = re.compile('[^ ㄱ - |가-힣]')
    result = hangul.sub('', text)
    # sub() 함수로 정규표현식을 hangul에 적용
    return result

    # 정규표현식 적용
    for i in tqdm(range(len(df))):
    df['Text'][i] = apply_regular_expression(df['Text'][i])
    """
)


st.write('3-2 불용어 제거')
st.text('불용어 사전을 사용하여 분석에 필요 없는 단어들을 제거')
st.code(
    """
    # 불용어 사전 로드 & 불용어 리스트 생성
    f = open('/content/drive/MyDrive/Stopwords.txt','r')
    lines = f.readlines()

    stop_words_list=[]

    for line in lines:
        stop_words_list.append(line.strip())
    """
)



st.write('3-3 형태소 분석')
st.text(
    """
    분석 속도가 가장 빠른 은전한닢(mecab) 형태소 분석기를 사용
    st.text('품사를 태깅(tagging)한 후 명사만 추출하여 분석에 사용
    """
)
st.code(
    """
    tokenizer = get_tokenizer('mecab')
    tokenized_sentence = []

    for sentence in tqdm(df['Text']):
        sent_mecab = tokenizer.nouns(sentence)
        # 형태소 분석기 적용 (명사)
        sent_mecab = [word for word in sent_mecab if not word in stop_words_list]
        # 불용어 제거
        sent_mecab = [word for word in sent_mecab if len(word) > 1]
        # 한글자 제거
        tokenized_word = []
        for word in sent_mecab:
            tokenized_word.append(word)
        # 리스트 안에 리스트 생성

        tokenized_sentence.append(tokenized_word)
        """
)

st.code(
    """
    # Countvectorizer 적용을 위해 list로 변환
    list_df = df['Text'].tolist()
    tokenizer = get_tokenizer('mecab')

    tokenized_sentence = []

    for sentence in tqdm(list_df):
        sent_mecab = tokenizer.nouns(sentence)
        # 형태소 분석기 적용 (명사)
        sent_mecab = [word for word in sent_mecab if not word in stop_words_list]
        # 불용어 제거
        sent_mecab = [word for word in sent_mecab if len(word) > 1]
        # 한글자 제거
        sent_mecab = (" ".join(sent_mecab)).strip()
        tokenized_sentence.append(sent_mecab)

    # CountVectorizer를 통한 토큰 코딩
    vect = CountVectorizer(max_features=10000, max_df=.10)
    X = vect.fit_transform(tokenized_sentence)
    """
)


st.markdown("***")

st.subheader("4) Word2vec 모델 학습")
st.markdown(
    """
    * 전처리된 학습 데이터로 Word2vec 모델 학습
    * Word2vec은 단어들을 embedding하여 단어 간 거리 계산 및 유사도를 파악 가능하게 함
    * Gensim 라이브러리 활용, 단어 벡터 차원은 100으로 설정
    * 100차원 상에 임베딩 된 단어는 총 8,998개에 해당
    """
)
st.code(
    """
    from gensim.models import Word2Vec
    model = Word2Vec(sentences = tokenized_sentence, size = 100, window = 5, min_count = 5, workers = 4, sg = 0)
    model.wv.vectors.shape
    """
)

st.markdown(
    """
    * 학습이 완료된 모델에 사전에 정의해두었던 Seedwords을 중심으로 단어 간 유사도를 파악
    * most_similar 함수는 지정한 단어와 가장 거리(유사도)가 가까운 단어 10개를 선정
    * 산출된 단어 리스트 중 다른 문화의 설명과 배타적이면서 범용적으로 사용되지 않는 단어들 선정
    * 선정한 단어로 표본 기업들의 조직 문화를 설명하는 단어 사전(기업 문화 사전)을 생성
    """
)

st.subheader("[기업 문화 사전]")
option_1 = st.selectbox('문화를 선택해주세요.',
                     ('혁신 문화', '시장 문화', '집단 문화', '위계 문화'))

if option_1 == '혁신 문화':
    st.write(option_1, ' : 창의, 창의력, 개성, 진취, 개방, 아이디어, 혁신, 역동, 토론, 도전, 모험, 시도, 자유, 프리, 자율, 플렉서블, 변화, 개혁, 속도, 탈피')
    from PIL import Image
    image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\사진파일\혁신문화.jpg")
    st.image(image)


if option_1 == '시장 문화':
    st.write(option_1, ' : 효율, 형식, 주먹구구식, 배분, 정형, 표준, 주먹구구, 도제식, 절차, 결재, 매뉴얼, 목표달성, 목표치, 채찍질, 달성, 신장, 실적, 압박감, 목표, 압박')
    from PIL import Image
    image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\사진파일\시장문화.jpg")
    st.image(image)


if option_1 == '집단 문화':
    st.write(option_1, ' : 가족, 화목, 편안, 젠틀, 육성, 양성, 유치, 적재적소, 강화, 배려, 존중, 단합, 유대감, 조화, 격려, 충성심, 충성, 신뢰, 헌신, 공감')
    from PIL import Image
    image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\사진파일\집단문화.jpg")
    st.image(image)


if option_1 == '위계 문화':
    st.write(option_1, ' : 권위, 복종, 올드, 위계질서, 위계, 상명, 계급, 강압, 위아래, 명령, 수직관, 상명하복, 의전, 보수, 경직, 폐쇄, 군대식, 수직, 군대, 질서, 엄격')
    from PIL import Image
    image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\사진파일\위계문화.jpg")
    st.image(image)

    
st.markdown("***")

st.subheader("5) 기업 문화 측정")

st.markdown(
    """
    * 분석 대상인 리뷰 수 상위 10개 기업의 리뷰 데이터를 추출
    * 단어-문서 발생 빈도(term frequency–inverse document frequency, TF-IDF)를 사용해 데이터 토큰화
    * TF-IDF를 기반으로 문서 단어 행렬(document-term matrix)을 형성
    * 기업 문화 사전을 사용해 문서 단어 행렬에서 각 문화 유형의 단어들의 TF-IDF 점수값을 합산,합산값이 그 기업이 보유한 문화 별 점수가 됨
    """
)

st.markdown("***")
st.subheader("6) 리뷰 수 상위 10개 기업 문화 시각화")

st.markdown(
    """
    * 기업 별 리뷰 내에서 Word2vec 모델로 생성된 기업 문화 사전 단어들의 TF-IDF 값 합산
    * Seaborn 라이브러리 사용
    * 리뷰 수 상위 10개 기업: 삼성전자㈜, LG전자㈜, LG디스플레이㈜, 현대자동차㈜, LG이노텍㈜, CJ제일제당㈜, 한국전력공사, LG화학㈜, 현대중공업㈜, SK하이닉스㈜
    """
)

st.markdown("***")
st.subheader("7) 기업 문화 측정치를 활용한 회귀분석")
st.info("https://colab.research.google.com/drive/1D4Hll797pitBNNE6XaM4_0C0iZ4lHZaE?usp=sharing")

st.markdown(
    """
    * 기업 문화 측정치와 연간 퇴사율,조직 만족도 간의 관계 측정
    * 기업 문화와 연간 퇴사율 간의 관계의 경우 포아송(Poisson) 회귀모형을 사용하여 관계성 측정
    * 기업 문화와 조직 만족도 간의 관계의 경우 잡플래닛의 별점 데이터를 숫자로 환산하여 분석에 활용, 다중회귀분석 이용
    """
)

st.write("종속변수: 연간 퇴사율")
st.code(
    """
    y = df['연간 퇴사율']
    X = df[['혁신 문화','시장 문화','집단 문화','위계 문화', '업력', '전체 평균연봉', '매출액', '조직 만족도']]
    X = sm.add_constant(X)

    model = GLM(y, X, family=Poisson())
    results = model.fit()
    results.summary()
    """
)
st.write("종속변수: 조직 만족도")
st.code(
    """
    y = df['조직 만족도']
    X = df[['혁신 문화','시장 문화','집단 문화','위계 문화', '업력', '전체 평균연봉', '매출액', '연간 입사율', '연간 퇴사율']]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    print(model.summary())
    """
)


st.markdown("***")
st.header("3. 분석 결과")
st.markdown(
    """
    본 연구에서 기업문화 측정치로 수행한 연구들은 다음과 같다. 
    (1) 리뷰 수 상위 10개 기업의 문화를 시각화 
    (2) 기업문화와 연간 이직율 간의 관계성, 기업문화와 조직 만족도 간의 관계성을 알아보는 회귀분석 시행 
    (3) 기업 수준 변수들을 활용한 연간 이직율과 조직 만족도 예측 모형 구축
    또한 본 연구에서는 기업 문화 측정치를 산출하기 전, 잡플래닛에서 크롤링한 리뷰 데이터들이 어떠한 주제와 단어들을 담고 있는지를 살펴보기 위해서 사전 조사로 워드 클라우드(Word Cloud)와 LDA 토픽모델링(Topic Modeling)을 활용하여 알아보았다
    """
)
st.subheader("1) Word Cloud")
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\워드클라우드.jpg")
st.image(image)

st.subheader("2) 토픽 모델링")
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\토픽모델링.jpg")
st.image(image)

st.subheader("3) 리뷰 수 상위 10개 기업 문화 시각화")
st.info(
    """
    리뷰 상위 10개 기업 중 첫번째로 삼성전자㈜는 혁신 문화가 약 34.2점으로 가장 높았으며 2번째로 높은 시장 문화(약 18.7점)과 10점 이상 차이나는 결과를 보여주었다. 이를 통해 삼성전자㈜는 두드러지는 혁신 문화를 보유했다고 판단할 수 있다.
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\samsung.jpg")
st.image(image)
st.info(
    """
    두번째로, LG전자㈜는 혁신 문화가 약 29.9점, 위계 문화가 21.7점으로 두 문화 모두 어느정도 공존하고 있는 양상을 보인다. 
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\lg.jpg")
st.image(image)
st.info(
    """
    LG디스플레이㈜는 위계 문화가 약 24.5점으로 가장 두드러졌으며, 두번째로 혁신 문화가 약 17.5점으로 뒤를 이었다.
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\lg display.jpg")
st.image(image)
st.info(
    """
    현대자동차㈜ 역시 위계 문화가 약 28.8점, 혁신 문화가 25.8점을 보이며 LG디스플레이㈜와 비슷한 양상을 보였다."""
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\hyundai.jpg")
st.image(image)
st. info(
    """
    LG이노텍㈜는 혁신 문화가 약 20.9점으로 가장 현저하였으며, 시장 문화가 약 12.4점으로 그 뒤를 이었다.
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\lg innotech.jpg")
st.image(image)
st.info(
    """
    CJ제일제당㈜의 경우, 시장 문화가 약 26.8점으로 가장 두드러졌으며, 그 뒤를 이어 혁신 문화가 23.5점, 위계 문화가 19.3점을 보였다.
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\CJ.jpg")
st.image(image)
st.info(
    """
    한국전력공사의 경우 위계 문화가 약 38.1점으로 다른 문화 유형에 비해 압도적으로 높게 측정되었다.
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\korean electricity.jpg")
st.image(image)
st. info(
    """
    LG화학㈜은 혁신 문화가 약 22.5점, 위계 문화가 약 22.3점으로 두 문화가 거의 비슷한 수준으로 공존하고 있었다.
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\lg chemistry.jpg")
st.image(image)
st.info(
    """
    현대중공업㈜는 위계 문화가 약 32.8점으로 가장 현저하게 높았고, 그 뒤를 이어 혁신 문화가 20.7점으로 측정되었다.
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\hyundai heavy industry.jpg")
st.image(image)
st.info(
    """
    SK하이닉스㈜의 경우, 혁신 문화가 약 20.2점, 위계 문화가 약 16점을 보였다.
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\사진파일\sk.jpg")
st.image(image)

st.info(
    """
    다음은 전체 100개 기업의 조직 문화 측정치의 평균을 계산하여 시각화한 것이다. 평균의 경우, 혁신 문화가 약 22.3점, 시장 문화가 약 14.5점, 집단 문화가 약 8.5점, 위계 문화가 21.4점을 보였다. 
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\조직문화 총기업.png")
st.image(image)
st.info(
    """
    이를 통해 알 수 있는 점은, 우리나라 제조/화학 산업군에 있는 기업들의 경우, 혁신과 위계 문화가 가장 높게 측정되는 양상을 보인다는 것이다. 보통의 경우, IT 산업과 같이 새로운 시도나 창의성이 중시되는 산업에서는 혁신 문화가, 생산성 증진과 비용절감이 중시되는 전통적인 제조업 산업에서는 시장이나 위계 문화가 현저할 것이라고 예측하는 것이 일반적이다. 그러나 본 연구 결과에 의하면 제조/화학 분야에서도 혁신 지향적인 문화가 위계 지향적인 문화 못지 않게 강조되고 있음을 볼 수 있다. 따라서 전통적인 제조업 분야에서도, 혁신의 중요성이 대두되고 있으며, 새로운 분야의 지식과 기술을 기존의 제품과 서비스에 접목시키는 역량이 크게 강조되고 있다고 해석할 여지가 있다.
    """
)

st.subheader("4) 회귀분석")
st.info(
    """
   기업문화와 연간 퇴사율 간의 관계
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\회귀분석_연간퇴사율.jpg")
st.image(image)

st.info(
    """
    기업문화와 조직 만족도 간의 관계
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\회귀분석_만족도.jpg")
st.image(image)

st.subheader("5) 예측 모형 결과")
st.markdown(
    """
    본 연구는 다중회귀분석에서 더 나아가 모델의 퇴사율 및 만족도를 예측하고자 잡플래닛의 기업데이터를 이용하여 예측 모델을 개발하였다. 
    퇴사율 모델을 개발하기 위해 퇴사율 및 만족도에 영향을 줄 수 있는 '혁신 문화', '시장 문화', '집단 문화', '위계 문화'를 독립변수로 투입하고 '업력', '인원','입사', '퇴사', '전체 평균연봉', '연간 입사자', '매출액' 변수 등을 통제변수로 넣었다. 
    변수의 종류에 따라 몇몇은 퇴사율 및 만족도과 양의 상관관계 또는 음의 상관관계가 확인되었다. 
    퇴사율을 예측한 다중선형회귀분석모델의 예측 결과의 성능은 0.40, RMSE는 24.21로 나타났다.
    만족도를 예측한 다중선형회귀분석모델의 예측 결과의 성능은 0.47, RMSE는 7.46로 나타났다. 
    퇴사율의 RMSE값이 높은 것은 250, 100등 이상치들이 포함되어 있었고, 학습데이터 수가 많지 않았기 때문으로 풀이된다. 
    """
)
st.info(
    """
    이직률 예측모형의 그래프
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\이직률 예측모형의 그래프.jpg")
st.image(image)

st.info(
    """
    이직률 실제 테스트 결과 모형의 그래프 
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\이직률 실제 테스트 결과 모형의 그래프.jpg")
st.image(image)

st.info(
    """
    만족도 예측모형의 그래프 
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\만족도 예측보형의 그래프.jpg")
st.image(image)

st.info(
    """
    만족도 실제 테스트 결과 모형의 그래프 
    """
)
from PIL import Image
image = Image.open("D:\문서\석사2학기\소셜네트워크 데이터마이닝과 분석\Final\contents\사진파일\만족도 실제 테스트 결과 모형의 그래프.jpg")
st.image(image)

st.markdown("***")

st.header("4. 연구 의의")
st.markdown("***")

st.header("5. 연구 한계점")
st.subheader("1) Word2vec 모델을 통한 기업 문화 분류의 타당성 제고")
st.markdown(
    """
    * Word2vec 모델을 통한 기업문화 분류가 타당한지 알아보기 위해 기존 문헌에서 사용되던 기업 문화 측정치와의 합치도를 확인해야 함
    """
)

st.subheader("2) 불용어 처리")
st.markdown(
    """
    * 불용어 처리를 최대한 시행하였으나, 분석 결과는 여전히 분석에 유의하지 않은 불용어를 많이 포함함
    """
)

st.subheader("3) 학습 데이터의 수")
st.markdown(
    """
    * 학습에 사용된 데이터는 85,010개였으나 Word2vec 모델에 embedding 된 단어는 8,917개 불과함
    * 따라서 더 타당성 있는 분석을 위해서는 적어도 10,000개 이상의 데이터가 필요함
    """
)

st.subheader('4) 기업 문화 사전 내 단어의 적절성')
st.markdown(
    """
    * TF-IDF를 통해 리뷰 내 범용적으로 사용되는 단어들의 중요도를 최대한 낮추었음
    * st.write('그러나 여전히 “결과“, “절차”, “목표“ 등 단어는 문화와 별개로 조직원들이 범용적으로 사용하고 있는 단어라는 점에서 적절성이 의심됨')
    """
)

st.markdown("***")

st.header("6. 향후 활용방안")

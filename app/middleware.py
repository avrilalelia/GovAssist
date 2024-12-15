from langchain.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
# import gradio as gr
import os

# Load environment variables
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
vectorStore = Chroma(persist_directory="../chroma", embedding_function=embeddings)

retriever = vectorStore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.5})

@tool
def progressBPJS(noReg: str) -> str:
    """See the status of specified registration number for application of BPJS Kesehatan

    Args:
        noReg: the registation number
    """
    url = f'{os.getenv("API_URL")}/bpjs/progress?noReg={noReg}'

    try:
        response = requests.get(url)
        if response.status_code == 200:
            posts = response.json()
            if posts and "status" in posts:
                    if posts["status"] == "approved":
                        return f'The status for your registration is {posts["status"]}. Go to the BPJS office at {posts["domisili"]} to verify your document.'
                    else:
                        return f'The status for your registration is {posts["status"]}.'
            else:
                return "There is no registration data with that registration number."
        else:
            print('Error:', response.status_code)
            return  response.status_code
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return e

@tool
def registrasiBPJS(nik: str, noKK: str, nama: str, domisili: str) -> str:
    """Input registration for BPJS Kesehatan

    Args:
        nik: No NIK
        noKK: No Kartu Keluarga/KK
        nama: Nama
        domisili: Domisili atau kota
    """
    url = f'{os.getenv("API_URL")}/bpjs/registrasi?nik={nik}&noKK={noKK}&nama={nama}&domisili={domisili}'

    try:
        response = requests.post(url)

        if response.status_code == 200:
            posts = response.json()
            regNo = posts["noReg"]
            return f"Registration is successful! Your registration number is {regNo}. Please wait for your registration status to be approved before verifying to your local BPJS office. You can check the status of your registration by asking me in this chat. Thank you."
        else:
            return "Sorry, there is something wrong. Please try again later"
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return "Sorry, there is something wrong. Please try again later"
    
@tool
def cek_akunBPJS(nik: str) -> str:
    """Check if user with specified NIK is registered for BPJS.

    Args:
        nik: No NIK
    """
    # Define the API endpoint URL
    url = f'{os.getenv("API_URL")}/bpjs/cek_akun?nik={nik}'

    try:
        response = requests.post(url)

        if response.status_code == 200:
            if response == None:
                return "Sorry, we cannot find your data. Please check if NIK is correct."
            posts = response.json()
            if posts:
                return f"You are registered for BPJS Kesehatan. Your health center is at ${posts["faskes"]}."

        else:
            return"Sorry, we cannot find your data. Please check if NIK is correct."
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return "Sorry, there is something wrong. Please try again later"
    
@tool
def progressKIP(noReg: str) -> str:
    """See the status of specified registration number for application of KIP

    Args:
        noReg: the registation number
    """
    url = f'{os.getenv("API_URL")}/kip/progress?noReg={noReg}'

    try:
        response = requests.get(url)
        if response.status_code == 200:
            posts = response.json()
            if posts and "status" in posts:
                    if posts["status"] == "approved":
                        return f'The status for your registration is {posts["status"]}. Go to the kip-kuliah.kemdikbud.go.id and login with your registered email.'
                    else:
                        return f'The status for your registration is {posts["status"]}.'
            else:
                return "There is no registration data with that registration number."
        else:
            print('Error:', response.status_code)
            return  response.status_code
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return e

@tool
def registrasiKIP(nik: str, nisn: str, npsn: str, email: str) -> str:
    """Input registration for KIP (Kartu Indonesia Pintar)

    Args:
        nik: No NIK
        nisn: Nomor Induk Siswa Nasional
        npsn: Nomor Pokok Sekolah Nasional
        email: Email
    """
    url = f'{os.getenv("API_URL")}/kip/registrasi?nik={nik}&nisn={nisn}&npsn={npsn}&email={email}'

    try:
        response = requests.post(url)

        if response.status_code == 200:
            posts = response.json()
            regNo = posts["noReg"]
            return f"Registration is successful! Your registration number is {regNo}. Please wait for your registration status to be approved before verifying to your local BPJS office. You can check the status of your registration by asking me in this chat. Thank you."
        else:
            return "Sorry, there is something wrong. Please try again later"
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return "Sorry, there is something wrong. Please try again later"
    
@tool
def cek_akunKIP(nik: str) -> str:
    """Check if user with specified NIK is registered as part of the KIP Program.

    Args:
        nik: No NIK
    """
    # Define the API endpoint URL
    url = f'{os.getenv("API_URL")}/kip/cek_akun?nik={nik}'

    try:
        response = requests.post(url)

        if response.status_code == 200:
            if response == None:
                return "Sorry, we cannot find your data. Please check if NIK is correct."
            posts = response.json()
            if posts:
                return f"You are registered for the KIP Program."

        else:
            return"Sorry, we cannot find your data. Please check if NIK is correct."
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return "Sorry, there is something wrong. Please try again later"
    
@tool
def progressPrakerja(noReg: str) -> str:
    """See the status of specified registration number for application of Prakerja

    Args:
        noReg: the registration number
    """
    url = f'{os.getenv("API_URL")}/prakerja/progress?noReg={noReg}'

    try:
        response = requests.get(url)
        if response.status_code == 200:
            posts = response.json()
            if posts and "status" in posts:
                    if posts["status"] == "approved":
                        return f'The status for your registration is {posts["status"]}. Go to https://www.prakerja.go.id and login with your registered email.'
                    else:
                        return f'The status for your registration is {posts["status"]}.'
            else:
                return "There is no registration data with that registration number."
        else:
            print('Error:', response.status_code)
            return  response.status_code
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return e

@tool
def registrasiPrakerja(nik: str, noKK: str, nama: str, domisili: str, no_hp: str, email: str) -> str:
    """Input registration for KIP (Kartu Indonesia Pintar)

    Args:
        nik: No NIK
        noKK: Nomor Kartu Keluarga (KK)
        nama: Nama
        domisili: Domisili
        no_hp: Nomor HP
        email: Email
    """
    url = f'{os.getenv("API_URL")}/prakerja/registrasi?nik={nik}&noKK={noKK}&nama={nama}&domisili={domisili}&no_hp={no_hp}&email={email}'

    try:
        response = requests.post(url)

        if response.status_code == 200:
            posts = response.json()
            regNo = posts["noReg"]
            return f"Registration is successful! Your registration number is {regNo}. Please wait for your registration status to be approved before verifying to your local BPJS office. You can check the status of your registration by asking me in this chat. Thank you."
        else:
            return "Sorry, there is something wrong. Please try again later"
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return "Sorry, there is something wrong. Please try again later"
    
@tool
def cek_akunPrakerja(nik: str) -> str:
    """Check if user with specified NIK is registered as part of the Prakerja Program.

    Args:
        nik: No NIK
    """
    # Define the API endpoint URL
    url = f'{os.getenv("API_URL")}/prakerja/cek_akun?nik={nik}'

    try:
        response = requests.post(url)

        if response.status_code == 200:
            if response == None:
                return "Sorry, we cannot find your data. Please check if NIK is correct."
            posts = response.json()
            if posts:
                return f"You are registered for the Prakerja Program."

        else:
            return"Sorry, we cannot find your data. Please check if NIK is correct."
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return "Sorry, there is something wrong. Please try again later"
    
@tool
def progressUMKM(noReg: str) -> str:
    """See the status of specified registration number for application of UMKM

    Args:
        noReg: the registration number
    """
    url = f'{os.getenv("API_URL")}/umkm/progress?noReg={noReg}'

    try:
        response = requests.get(url)
        if response.status_code == 200:
            posts = response.json()
            if posts and "status" in posts:
                    if posts["status"] == "approved":
                        return f'The status for your registration is {posts["status"]}. Go to https://oss.go.id/ and login with your registered email.'
                    else:
                        return f'The status for your registration is {posts["status"]}.'
            else:
                return "There is no registration data with that registration number."
        else:
            print('Error:', response.status_code)
            return  response.status_code
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return e

@tool
def registrasiUMKM(nik: str, noKK: str, nama: str, domisili: str, nama_usaha: str, skala_bisnis: str, no_hp: str, email: str) -> str:
    """Input registration for UMKM program

    Args:
        nik: No NIK
        noKK: Nomor Kartu Keluarga (KK)
        nama: Nama
        domisili: Domisili
        nama_usaha: Nama Usaha
        skala_bisnis: Skala Bisnis
        no_hp: Nomor HP
        email: Email
    """
    url = f'{os.getenv("API_URL")}/umkm/registrasi?nik={nik}&noKK={noKK}&nama={nama}&domisili={domisili}&no_hp={no_hp}&email={email}&nama_usaha={nama_usaha}&skala_bisnis={skala_bisnis}'

    try:
        response = requests.post(url)

        if response.status_code == 200:
            posts = response.json()
            regNo = posts["noReg"]
            return f"Registration is successful! Your registration number is {regNo}. Please wait for your registration status to be approved before verification step at UMKM website. You can check the status of your registration by asking me in this chat. Thank you."
        else:
            return "Sorry, there is something wrong. Please try again later"
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return "Sorry, there is something wrong. Please try again later"
    
@tool
def cek_akunUMKM(nik: str) -> str:
    """Check if user with specified NIK is registered as part of the UMKM Program.

    Args:
        nik: No NIK
    """
    # Define the API endpoint URL
    url = f'{os.getenv("API_URL")}/umkm/cek_akun?nik={nik}'

    try:
        response = requests.post(url)

        if response.status_code == 200:
            if response == None:
                return "Sorry, we cannot find your data. Please check if NIK is correct."
            posts = response.json()
            if posts:
                return f"You are registered for the UMKM Program."

        else:
            return"Sorry, we cannot find your data. Please check if NIK is correct."
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return "Sorry, there is something wrong. Please try again later"
    
@tool
def progressBansos(noReg: str) -> str:
    """See the status of specified registration number for application of Bansos (Bantuan Sosial)

    Args:
        noReg: the registration number
    """
    url = f'{os.getenv("API_URL")}/bansos/progress?noReg={noReg}'

    try:
        response = requests.get(url)
        if response.status_code == 200:
            posts = response.json()
            if posts and "status" in posts:
                    if posts["status"] == "approved":
                        return f'The status for your registration is {posts["status"]}. Install Cek Bansos App and login with your registered email.'
                    else:
                        return f'The status for your registration is {posts["status"]}.'
            else:
                return "There is no registration data with that registration number."
        else:
            print('Error:', response.status_code)
            return  response.status_code
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return e

@tool
def registrasiBansos(nik: str, noKK: str, nama: str, domisili: str) -> str:
    """Input registration for Bansos (Bantuan Sosial) program

    Args:
        nik: No NIK
        noKK: Nomor Kartu Keluarga (KK)
        nama: Nama
        domisili: Domisili
    """
    url = f'{os.getenv("API_URL")}/bansos/registrasi?nik={nik}&noKK={noKK}&nama={nama}&domisili={domisili}'

    try:
        response = requests.post(url)

        if response.status_code == 200:
            posts = response.json()
            regNo = posts["noReg"]
            return f"Registration is successful! Your registration number is {regNo}. Please wait for your registration status to be approved before verification step. You can check the status of your registration by asking me in this chat. Thank you."
        else:
            return "Sorry, there is something wrong. Please try again later"
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return "Sorry, there is something wrong. Please try again later"
    
@tool
def cek_akunBansos(nik: str) -> str:
    """Check if user with specified NIK is registered as part of the Bansos (Bantuan Sosial) Program.

    Args:
        nik: No NIK
    """
    # Define the API endpoint URL
    url = f'{os.getenv("API_URL")}/bansos/cek_akun?nik={nik}'

    try:
        response = requests.post(url)

        if response.status_code == 200:
            if response == None:
                return "Sorry, we cannot find your data. Please check if NIK is correct."
            posts = response.json()
            if posts:
                return f"You are registered for the Bansos Program."

        else:
            return"Sorry, we cannot find your data. Please check if NIK is correct."
        
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return "Sorry, there is something wrong. Please try again later"
    
tools = [progressBPJS, registrasiBPJS, cek_akunBPJS, progressKIP, registrasiKIP, cek_akunKIP, progressPrakerja, registrasiPrakerja, cek_akunPrakerja, progressUMKM, registrasiUMKM, cek_akunUMKM, progressBansos, registrasiBansos, cek_akunBansos]

workflow = StateGraph(state_schema=MessagesState)
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3, max_tokens=500, api_key=os.getenv("GOOGLE_API_KEY"), streaming=True)
model = model.bind_tools(tools)

# Define the function that calls the model
def call_model(state: MessagesState):
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    human_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage): 
            human_message = message.content
            break
    
    if human_message:
        retrieved_docs = retriever.get_relevant_documents(human_message)
        print(retrieved_docs)
        
        if retrieved_docs:
            context_message = "\n".join([doc.page_content for doc in retrieved_docs])
            messages = [SystemMessage(content=context_message)] + state["messages"]

    response = model.invoke(messages)
    return {"messages": response}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
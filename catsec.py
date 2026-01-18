from typing import List
from langchain_core.documents import Document

from cat import tool, hook, plugin, AgenticWorkflowTask, RecallSettings, UserMessage
from cat.log import log
import requests
from pydantic import BaseModel, field_validator
import pickle
from datetime import datetime, timedelta

PICKLE_FILE = '/app/cat/data/catnesina_updates.pkl'


def validate_threshold(value):
    if value < 0 or value > 1:
        return False

    return True


class MySettings(BaseModel):
    k: int = 10
    threshold: float = 0.4
    last_n_history_messages: int = 10
    chunk_size: int = 256
    chunk_overlap: int = 128
    document_expiration_in_days: int = 1

    @field_validator("threshold")
    @classmethod
    def threshold_validator(cls, threshold):
        if not validate_threshold(threshold):
            raise ValueError("Declarative memory threshold must be between 0 and 1")

    @field_validator("document_expiration_in_days")
    @classmethod
    def document_expiration_in_days_threshold_validator(cls, threshold):
        if 1 >= threshold >= 31:
            raise ValueError("Document expiration threshold must be between 1 and 30")

@plugin
def settings_model():
    return MySettings


mappatura_iso = {
    "Afghanistan": "AFG",
    "Albania": "ALB",
    "Algeria": "DZA",
    "Andorra": "AND",
    "Angola": "AGO",
    "Anguilla": "AIA",
    "Antigua e Barbuda": "ATG",
    "Arabia Saudita": "SAU",
    "Argentina": "ARG",
    "Armenia": "ARM",
    "Aruba": "ABW",
    "Australia": "AUS",
    "Austria": "AUT",
    "Azerbaigian": "AZE",
    "Bahamas": "BHS",
    "Bahrein": "BHR",
    "Bangladesh": "BGD",
    "Barbados": "BRB",
    "Belgio": "BEL",
    "Belize": "BLZ",
    "Benin": "BEN",
    "Bermuda": "BMU",
    "Bhutan": "BTN",
    "Bielorussia": "BLR",
    "Bolivia": "BOL",
    "Bosnia-Erzegovina": "BIH",
    "Botswana": "BWA",
    "Brasile": "BRA",
    "Brunei": "BRN",
    "Bulgaria": "BGR",
    "Burkina Faso": "BFA",
    "Burundi": "BDI",
    "Cambogia": "KHM",
    "Camerun": "CMR",
    "Canada": "CAN",
    "Capo Verde": "CPV",
    "Ciad": "TCD",
    "Cile": "CHL",
    "Cipro": "CYP",
    "Colombia": "COL",
    "Comore": "COM",
    "Congo": "COG",
    "Costa d'Avorio": "CIV",
    "Costa Rica": "CRI",
    "Croazia": "HRV",
    "Cuba": "CUB",
    "Curacao": "CUW",
    "Danimarca": "DNK",
    "Dominica": "DMA",
    "Ecuador": "ECU",
    "Egitto": "EGY",
    "El Salvador": "SLV",
    "Emirati Arabi Uniti": "ARE",
    "Eritrea": "ERI",
    "Estonia": "EST",
    "Etiopia": "ETH",
    "Federazione Russa": "RUS",
    "Figi": "FJI",
    "Filippine": "PHL",
    "Finlandia": "FIN",
    "Francia": "FRA",
    "Gabon": "GAB",
    "Gambia": "GMB",
    "Georgia": "GEO",
    "Germania": "DEU",
    "Ghana": "GHA",
    "Giamaica": "JAM",
    "Giappone": "JPN",
    "Gibilterra": "GIB",
    "Gibuti": "DJI",
    "Giordania": "JOR",
    "Grecia": "GRC",
    "Grenada": "GRD",
    "Guadalupa": "GLP",
    "Guam": "GUM",
    "Guatemala": "GTM",
    "Guinea": "GIN",
    "Guinea-Bissau": "GNB",
    "Guinea Equatoriale": "GNQ",
    "Guyana": "GUY",
    "Guyana Francese": "GUF",
    "Haiti": "HTI",
    "Honduras": "HND",
    "Hong Kong": "HKG",
    "India": "IND",
    "Indonesia": "IDN",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Irlanda": "IRL",
    "Islanda": "ISL",
    "Isole BES": "BES",
    "Isole Cayman": "CYM",
    "Isole Cook": "COK",
    "Isole Marianne Settentrionali": "MNP",
    "Isole Marshall": "MHL",
    "Isole Salomone": "SLB",
    "Isole Turks e Caicos": "TCA",
    "Isole Vergini Americane": "VIR",
    "Isole Vergini Britanniche": "VGB",
    "Israele": "ISR",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Kirghizistan": "KGZ",
    "Kiribati": "KIR",
    "Kosovo": "KSV",
    "Kuwait": "KWT",
    "Laos": "LAO",
    "Lesotho": "LSO",
    "Lettonia": "LVA",
    "Libano": "LBN",
    "Liberia": "LBR",
    "Libia": "LBY",
    "Liechtenstein": "LIE",
    "Lituania": "LTU",
    "Lussemburgo": "LUX",
    "Macao": "MAC",
    "Madagascar": "MDG",
    "Malawi": "MWI",
    "Malaysia": "MYS",
    "Maldive": "MDV",
    "Mali": "MLI",
    "Malta": "MLT",
    "Marocco": "MAR",
    "Martinica": "MTQ",
    "Mauritania": "MRT",
    "Mauritius": "MUS",
    "Mayotte": "MYT",
    "Messico": "MEX",
    "Repubblica di Moldova": "MDA",
    "Monaco": "MCO",
    "Mongolia": "MNG",
    "Montenegro": "MNE",
    "Montserrat": "MSR",
    "Mozambico": "MOZ",
    "Myanmar": "MMR",
    "Namibia": "NAM",
    "Nauru": "NRU",
    "Nepal": "NPL",
    "Nicaragua": "NIC",
    "Niger": "NER",
    "Nigeria": "NGA",
    "Niue": "NIU",
    "Norvegia": "NOR",
    "Nuova Caledonia": "NCL",
    "Nuova Zelanda": "NZL",
    "Oman": "OMN",
    "Paesi Bassi": "NLD",
    "Pakistan": "PAK",
    "Panama": "PAN",
    "Papua Nuova Guinea": "PNG",
    "Paraguay": "PRY",
    "Perù": "PER",
    "Polinesia Francese": "PYF",
    "Polonia": "POL",
    "Portogallo": "PRT",
    "Qatar": "QAT",
    "Regno di Eswatini": "SWZ",
    "Regno Unito": "GBR",
    "Repubblica Ceca": "CZE",
    "Repubblica Centrafricana": "CAF",
    "Repubblica Democratica del Congo": "COD",
    "Repubblica Democratica di Timor Est": "TLS",
    "Repubblica di Corea (Corea del Sud)": "KOR",
    "Repubblica di Macedonia del Nord": "MKD",
    "Repubblica di Palau": "PLW",
    "Repubblica di Serbia": "SRB",
    "Repubblica Dominicana": "DOM",
    "Repubblica Popolare Cinese": "CHN",
    "Repubblica Popolare Democratica di Corea": "PRK",
    "Reunion": "REU",
    "Romania": "ROU",
    "Ruanda": "RWA",
    "Saint Kitts e Nevis": "KNA",
    "Saint Lucia": "LCA",
    "Saint Vincent e Grenadine": "VCT",
    "Saint-Martin": "MAF",
    "Samoa": "WSM",
    "Samoa Americane": "ASM",
    "San Marino": "SMR",
    "Sao Tomé e Principe": "STP",
    "Senegal": "SEN",
    "Seychelles": "SYC",
    "Sierra Leone": "SLE",
    "Singapore": "SGP",
    "Sint Maarten": "SXM",
    "Siria": "SYR",
    "Slovacchia": "SVK",
    "Slovenia": "SVN",
    "Somalia": "SOM",
    "Spagna": "ESP",
    "Sri Lanka": "LKA",
    "Stati Federati di Micronesia": "FSM",
    "Stati Uniti d'America": "USA",
    "Sudafrica": "ZAF",
    "Sud Sudan": "SSD",
    "Sudan": "SDN",
    "Suriname": "SUR",
    "Svezia": "SWE",
    "Svizzera": "CHE",
    "Tagikistan": "TJK",
    "Taiwan": "TWN",
    "Tanzania": "TZA",
    "Territori Palestinesi": "PSE",
    "Thailandia": "THA",
    "Togo": "TGO",
    "Tonga": "TON",
    "Trinidad e Tobago": "TTO",
    "Tunisia": "TUN",
    "Turchia": "TUR",
    "Turkmenistan": "TKM",
    "Tuvalu": "TUV",
    "Ucraina": "UKR",
    "Uganda": "UGA",
    "Ungheria": "HUN",
    "Uruguay": "URY",
    "Uzbekistan": "UZB",
    "Vanuatu": "VUT",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
    "Yemen": "YEM",
    "Zambia": "ZMB",
    "Zimbabwe": "ZWE"
    }


@hook
def before_cat_reads_message(user_message: UserMessage, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    last_n_history_messages = settings.last_n_history_messages

    conversation_so_far = [
        f"{history_item.who}: {history_item.content.text}"
        for history_item in cat.working_memory.history[-last_n_history_messages:]
    ]
    conversation_so_far = "\n".join(conversation_so_far)

    prompt = f"""Here is a conversation between an AI and a Human:
{conversation_so_far}

Sum up in a single sentence what the human wants in their own language, keeping all the details from the conversation.
The sentence must be expressed from the point of view of the human. Beware of topic changes, if the topic (i.e. the city/country) changes then only compress the latest topic (city/country) conversation.

Sentence: """

    # prepare agent input
    output = cat.agentic_workflow.run(
        task=AgenticWorkflowTask(
            system_prompt=prompt,
            user_prompt=user_message.text,
        ),
        llm=cat.large_language_model,
    )

    return output.output


def load_updates():
    try:
        with open(PICKLE_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


updated_countries = load_updates()


def save_updates():
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(updated_countries, f)


def is_older_than_1_day(stored_datetime):
    value = stored_datetime < datetime.now() - timedelta(days=1)
    print(f"VALUE: {value}")
    return value


@tool
def get_country_report(tool_input, cat):
    """"
    Use this function whenever the user asks questions about vaccinations, consulates and embassies, visas, security,
    terrorism, environmental risks, local mobility, local laws, customs, and currency formalities in a country or in a
    city.
    This function is useful to download the security and risks' country report from the right webpage.
    The input of this function is the country name, the only allowed country names are in the following list, ensure to
    use exactly the allowed string: e.g. ['Afghanistan', 'Albania',, ...]. The output is always None.
    """
    if tool_input in updated_countries and not is_older_than_1_day(updated_countries[tool_input]):
        return None

    cat.send_ws_message(
        msg_type="chat",
        content=f"Sto andando a verificare sul sito della Farnesina le ultime informazioni disponibili. Attendi un momento.",
    )

    url_pdf = f"https://www.viaggiaresicuri.it/schede_paese/pdf/{mappatura_iso[tool_input]}.pdf"
    download_path = f'/app/cat/plugins/catnesina/downloads/{tool_input}.pdf'

    res = requests.get(url_pdf)
    if res.status_code == 200:
        with open(download_path, "wb") as file:
            file.write(res.content)
            print(f"{tool_input}.pdf downloaded with success")
        
        cat.rabbit_hole.ingest_file(cat, download_path)
        updated_countries[tool_input] = datetime.now()
        save_updates()
    else:
        print(f"Something was wrong, viaggiaresicuri.it returned HTTP {res.status_code}")

    print(f"################ {tool_input}")
    return tool_input


@hook
def agent_prompt_prefix(prefix, cat):
    return """Sei un consulente di un'importante società.
Il tuo è un alto profilo orientato al business e alla compliance.
Il tuo linguaggio è formale, e utilizzi gli elenchi puntati o numerati solo se strettamente necessario per la comprensione, altrimenti preferisci una forma discorsiva.
Le tue risposte sono sempre complete, con attenzione al business, alla sicurezza fisica, agli obblighi di legge e alla compliance.
"""


@hook
def agent_prompt_suffix(suffix, cat):
    return """ALWAYS answer in the same language used by the user. Call the user 'Viaggiatore'.
# Context

{context}
"""


@hook
def before_cat_recalls_memories(config: RecallSettings, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    config.k = settings["k"]
    config.threshold = settings["threshold"]

    return config


@hook
def before_rabbithole_stores_documents(docs: List[Document], cat) -> List[Document]:
    group_size = 5

    notification = f"Starting to summarize {len(docs)}",
    log(notification, "INFO")
    cat.send_ws_message(notification, msg_type="notification")

    # we will store iterative summaries all together in a list
    all_summaries = []

    # Compute total summaries for progress notification
    n_summaries = len(docs) // group_size

    # make summaries of groups of docs
    for n, i in enumerate(range(0, len(docs), group_size)):
        # Notify the admin of the progress
        progress = (n * 100) // n_summaries
        message = f"{progress}% of summarization"
        cat.send_ws_message(message, msg_type="notification")
        log(message, "INFO")

        # Get the text from groups of docs and join to string
        group = docs[i: i + group_size]
        group = list(map(lambda d: d.page_content, group))
        text_to_summarize = "\n".join(group)

        # prepare agent input
        output = cat.agentic_workflow.run(
            task=AgenticWorkflowTask(user_prompt=f"Write a concise summary of the following: {text_to_summarize}"),
            llm=cat.large_language_model,
        )
        # Summarize and add metadata
        summary = Document(page_content=output.output)
        summary.metadata["is_summary"] = True

        # add the new summary to a list of all summaries
        all_summaries.append(summary)

    docs.extend(all_summaries)

    return docs

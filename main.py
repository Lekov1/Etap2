import os

from huggingface_hub import login
login(token=os.environ["HUG_TOKEN"])

from llms.llms.langchain_wrappers.chat_together import ChatTogether

model = ChatTogether(
    model="deepseek-ai/deepseek-coder-33b-instruct",
    temperature=0.2
)

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

chain = model | parser

from langchain_core.prompts import ChatPromptTemplate

system_template = " You are a professional translator. Please translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

chain = prompt_template | model | parser

language= "English"
text= """"
Günümüzün hızla değişen iş dünyasında, şirketlerin başarıya ulaşmak için sürekli olarak kendilerini geliştirmeleri ve yeniliklere ayak uydurmaları gerekmektedir. Rekabet ortamının giderek arttığı bu dönemde, işletmelerin hayatta kalabilmeleri için müşteri odaklı bir yaklaşım benimsemeleri ve müşteri memnuniyetini en üst düzeyde tutmaları oldukça önemlidir.
Müşteri memnuniyetini sağlamanın en etkili yollarından biri, sunulan ürün ve hizmetlerin kalitesini artırmaktır. Bunun için, şirketlerin ar-ge çalışmalarına yatırım yapmaları, yenilikçi ürünler geliştirmeleri ve hizmet kalitelerini sürekli olarak iyileştirmeleri gerekmektedir. Aynı zamanda, müşterilerin değişen ihtiyaç ve beklentilerini yakından takip etmek ve bu doğrultuda çözümler üretmek de büyük önem taşımaktadır.
Şirketlerin başarısında rol oynayan bir diğer faktör ise insan kaynağıdır. Nitelikli, motivasyonu yüksek ve işine bağlı çalışanlar, işletmelerin hedeflerine ulaşmasında kilit rol oynamaktadır. Bu nedenle, şirketlerin doğru yetenekleri bünyelerine katmaları, çalışanlarının gelişimine yatırım yapmaları ve onları motive edecek bir çalışma ortamı yaratmaları son derece önemlidir.
Tüm bunların yanı sıra, günümüzün dijital çağında teknolojinin sunduğu imkanlardan etkin bir şekilde yararlanmak da şirketlerin rekabet avantajı elde etmesinde büyük rol oynamaktadır. Dijital dönüşüm sürecine ayak uyduran, iş süreçlerini otomatize eden ve veri analitiği gibi araçları etkin bir şekilde kullanan şirketler, rakiplerinin bir adım önüne geçme fırsatı yakalamaktadır.
Sonuç olarak, değişen dünyaya uyum sağlayabilen, müşteri odaklı bir yaklaşım benimseyen, insan kaynağına yatırım yapan ve teknolojiyi etkin bir şekilde kullanan şirketler, uzun vadede başarıya ulaşma konusunda önemli bir avantaj elde etmektedirler. Bu doğrultuda atılacak stratejik adımlar ve sürdürülebilir bir gelişim anlayışı, şirketlerin geleceğini şekillendirmede kilit rol oynayacaktır.
"""

result = chain.invoke({"language": language, "text": text})

model_name = model.model_name.replace("/", "_")

# Write the result to a file with the model name
with open(f"{model_name}.txt", "w", encoding="utf-8") as f:
    f.write(result)

print(result)
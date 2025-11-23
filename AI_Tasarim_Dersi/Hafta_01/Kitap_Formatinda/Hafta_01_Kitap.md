# Bölüm 1: Fikirden Görsele - Yapay Zeka ile Yaratıcılığın Başlangıcı

**(Tahmini Okuma Süresi: Yaklaşık 1.5 - 2 saat)**

## Giriş ve Teorik Temeller

Tasarım dünyasında yepyeni bir çağın eşiğindeyiz. Üretken yapay zeka (Generative AI), artık yalnızca bir araç olmaktan çıkıp, yaratıcılığımızın sınırlarını zorlayan, fikirlerimizi saniyeler içinde görselleştirmemizi sağlayan bir iş arkadaşı haline geliyor. Bu ilk bölümde, bu devrimin temellerini atacak, yapay zekanın sunduğu muazzam potansiyeli keşfedecek ve hayalimizdeki konseptleri somut görsellere dönüştürmenin ilk adımlarını öğreneceğiz. Bu dönüşüm, tasarımcı rolünü de temelden etkilemektedir.

### 1.1. Üretken Yapay Zeka (Generative AI): Tanımı ve Tasarım Alanındaki Rolü

Üretken Yapay Zeka (Generative AI), mevcut verilerden öğrenerek daha önce var olmayan özgün içerikler üretebilen bir yapay zeka dalıdır. Bu içerikler metin, görsel, ses, kod veya diğer veri türleri olabilir. Temelinde, büyük veri kümelerindeki kalıpları, ilişkileri ve yapıları öğrenen algoritmalar bulunur. Yapay zeka modelleri, bu öğrendiklerini yeni girdilere (prompt'lar gibi) uygulayarak, olasılıksal olarak en uygun ve yaratıcı çıktıyı oluşturur.

Geleneksel olarak tasarımcılar, fırçayı, kalemi, yazılımı birer araç olarak kullanırdı. Ancak üretken yapay zeka, bu ilişkiyi kökten değiştiriyor. Yapay zekayı sadece bir "Photoshop filtresi" veya basit bir otomasyon aracı olarak görmek yerine, yaratıcılığımızı tetikleyen, bize beklenmedik fikirler sunan, hatta bir "dijital beyin fırtınası ortağı" olarak konumlandırmak, bu yeni dönemin anahtarıdır. Yapay zeka, tasarım sürecindeki tekrarlayan ve zaman alan görevleri (örneğin, temel kompozisyonlar oluşturma, varyasyonlar üretme) üstlenerek tasarımcının daha stratejik ve yaratıcı düşünmeye odaklanmasını sağlar. Tasarımcı rolümüz, artık sadece "nasıl yapacağını" bilen kişi olmaktan çıkıp, "ne yapılacağını" belirleyen, vizyonu yönlendiren, AI'dan çıkan sonuçları en doğru şekilde seçip düzenleyen ve projenin nihai estetik ve fonksiyonel hedeflerini güvence altına alan bir "yaratıcı yönetmen" veya "küratör" rolüne evriliyor. Bu, insan yaratıcılığının yerini almak değil, onu inanılmaz derecede güçlendirmek ve tasarımın kapsamını genişletmek anlamına gelir.

**(Görsel Önerisi: Bir tarafta klasik bir tasarımcı masası (eskizler, kalemler, büyük monitörler), diğer tarafta ise bir insanın modern bir arayüzle (belki tablet veya laptop) yapay zeka ile etkileşimde olduğu, soyut ve stilize bir görsel. Bu görsel, geleneksel araçlar ve AI destekli yeni çalışma biçimleri arasındaki geçişi temsil edebilir.)**

### 1.2. Yapay Zeka ile Etkili İletişim: Prompt Mühendisliği'nin Önemi

AI ile etkileşim kurarken aslında onunla bir "diyalog" kuruyoruz. Basit bir komutla elde edeceğimiz sonuç ile, onu detaylı bir şekilde tarif ederek elde edeceğimiz sonuç arasında devasa bir fark olacaktır. Bu, bir yönetmenin sahneyi oyuncuya tarif etmesi gibidir; ne kadar net ve tutkulu tarif ederseniz, oyuncu o kadar istediğiniz performansı sergiler. Prompt yazmak, bir nevi yapay zekanın hayal gücünü yönlendirme sanatıdır. Ancak prompt mühendisliği sadece AI ile "konuşmak" anlamına gelmez; aynı zamanda AI'nın çıktısını *kalite kontrol etmek*, *beklenmedik veya hatalı sonuçları yönetmek* ve *yaratıcı süreci istenen yönde ilerletmek* için kritik bir beceridir. Etkili prompt'lar sayesinde, AI'nın potansiyelinden tam olarak faydalanabilir, tekrarlayan deneme-yanılma döngülerini azaltabilir ve projenizin estetik ve fonksiyonel hedeflerine daha hızlı ulaşabiliriz.

#### 1.2.1. Yapay Zeka ile Nasıl Konuşacağız? Temel Kavramlar

Bu yeni dilin temel taşlarını anlamak, etkili komutlar yazmamızı sağlar:

*   **Model:** Yapay zekanın "beyni" veya "sanatçısı" diyebiliriz. Her model (örn: Midjourney v6, DALL-E 3, Adobe Firefly) farklı bir eğitim setinden geçmiş ve kendine özgü bir "sanat anlayışına" sahiptir. Bir model daha gerçekçi görseller üretirken, diğeri daha soyut veya fantastik olabilir. Bir ressamın farklı stillerde çalışabilmesi gibi düşünebiliriz. Model seçimi, elde edilecek görselin temel karakterini belirler.
*   **Prompt (Komut):** Yapay zekaya ne istediğimizi anlatan metin komutudur. Tıpkı bir yönetmenin oyuncuya sahneyi tarif etmesi gibi, ne kadar detaylı ve net olursanız, sonuç o kadar istediğinize yakın olur. Bu, AI ile kurduğumuz temel iletişim dilidir ve görselin içeriğini, stilini, atmosferini belirleyen ana unsurdur.
*   **Parametre:** Prompt'un kendisi dışında modele verdiğimiz teknik komutlar veya ayarlar. Bunlar, görselin en-boy oranı (`--ar 16:9`), stil referansı (`--style raw`) veya belirli bir rastgelelik seviyesi (`--chaos 50`) gibi, çıktının stilini ve yapısını ince ayarlamamızı sağlar. Yönetmenin oyuncuya "daha dramatik oyna" gibi ek talimatlar vermesi gibidir. Parametreler, AI'nın üreteceği görselin kompozisyonu, detay seviyesi ve estetik eğilimleri üzerinde ince ayarlar yapmamıza olanak tanır.
*   **Render (Oluşturma):** Modelin prompt'u alıp onu görsel bir çıktıya dönüştürme işlemidir. Bu işlem, modelin karmaşıklığına ve sunucu yoğunluğuna göre saniyeler sürebilir. Bir sanatçının sizin tarifinizi alıp tabloyu tamamlaması gibi düşünebilirsiniz. Render süreci, yapay zekanın "yaratım anıdır" ve elde edilen sonuç prompt'un ve parametrelerin doğruluğunu test etme imkanı sunar.

**(Görsel Önerisi: Bu kavramları simgeleyen, akılda kalıcı ve basit ikonlar. Örneğin: Beyin ikonu (Model), Konuşma balonu ikonu (Prompt), Ayarlar dişlisi ikonu (Parametre), Fotoğraf makinesi ikonu (Render). Her ikonun yanına kısa, akademik bir açıklama eklenebilir.)**

#### 1.2.2. Kapsamlı Prompt Formülü: Adım Adım Görsel Tarifi

En iyi sonuçlar genellikle belirli bir yapıyı takip eden, detaylı prompt'lardan gelir. Bu formül, AI'nın ne istediğinizi daha iyi anlamasına ve istediğiniz görseli üretmesine yardımcı olur. Aklınızdaki görseli şu adımlarla tarif etmeyi deneyin:

*   **`[GÖRSEL TÜRÜ]`**: Ne tür bir çıktı istediğinizi belirtin (örn: `photo`, `illustration`, `3d render`, `vector logo`, `watercolor painting`, `line art`). Bu, AI'nın doğru "sanatçı" modunu seçmesine yardımcı olur.
    *   *Örnek: "photo" (fotoğraf)*
*   **`[ANA KONU/NESNE]`**: Görselin ana odak noktası nedir? (örn: `a sleeping cat`, `a futuristic city`, `a steaming cup of coffee`).
    *   *Örnek: "a sleeping ginger tabby cat" (uyuyan zencefil rengi tekir kedi)*
*   **`[DETAYLAR/SIFATLAR]`**: Ana konuyu daha belirgin hale getiren tanımlayıcılar. (örn: `fluffy and cute`, `ancient and mystical`, `aromatic and rich`).
    *   *Örnek: "fluffy and cute" (tüylü ve sevimli)*
*   **`[ORTAM/ARKA PLAN]`**: Konunun nerede bulunduğunu tarif edin. (örn: `on a stack of old books`, `in a cozy library`, `on a wooden table`, `in a misty forest`, `with a blurry background`).
    *   *Örnek: "on a stack of old books, in a cozy library" (eski bir kitap yığınının üzerinde, rahat bir kütüphanede)*
*   **`[STİL/SANAT AKIMI]`**: Görselin genel estetik anlayışı. (örn: `style of fantasy art`, `minimalist`, `cyberpunk`, `vintage`, `studio ghibli style`, `impressionism`). Bu, görselin sanatsal ve kalitesini doğrudan etkiler.
    *   *Örnek: "style of fantasy art" (fantastik sanat tarzında)*
*   **`[KOMPOZİSYON/KAMERA AÇISI]`**: Sahnenin nasıl çerçevelendiği. (örn: `close-up shot`, `wide-angle shot`, `from above, bird's-eye view`, `cinematic`). Bu, görselin anlatısal gücünü ve izleyici üzerindeki etkisini artırır.
    *   *Örnek: "close-up shot" (yakın çekim)*
*   **`[RENK/IŞIKLANDIRMA]`**: Görselin atmosferini ve modunu belirler. (örn: `warm, dramatic lighting`, `soft pastel colors`, `neon glow`, `vibrant colors`). Renk ve ışıklandırma, duygu durumunu ve mesajı doğrudan iletir.
    *   *Örnek: "warm, dramatic lighting" (sıcak, dramatik aydınlatma)*

**Örnek Uygulama:**
*   **Basit:** `A cat` -> Sadece bir kedi görseli. Genellikle genel ve beklentiden uzak sonuçlar verir.
*   **Daha İyi:** `A sleeping ginger cat` -> Uyuyan, turuncu bir kedi. Konu ve temel bir eylem belirtildiği için daha odaklı bir sonuç alınır.
*   **Harika (Yukarıdaki formülle oluşturulmuş):** `**photo** of a **sleeping ginger tabby cat**, **fluffy and cute**, **on a stack of old books**, **in a cozy library**, **style of fantasy art**, **close-up shot**, **warm, dramatic lighting` -> Detaylı, atmosferik ve istenen tarza uygun bir görsel elde edilir. Bu, AI'nın belirttiğiniz tüm parametreleri hesaba katmasını sağlar.

#### 1.2.3. Prompt Örnekleri: Farklı Senaryolar ve Stiller

Aşağıda, farklı senaryolar için kullanabileceğiniz ve AI'nın farklı yeteneklerini ortaya çıkaracak prompt örneklerini bulabilirsiniz. Bu örnekler, hayal gücünüzü tetiklemek ve ne kadar çeşitli çıktılar alabileceğinizi göstermek için hazırlanmıştır. Her bir prompt, belirli bir amaç ve estetik hedefe yönelik olarak tasarlanmıştır.

*   **Ürün Odaklı (Minimalist Kulaklık Kılıfı):**
    `product photography of a sleek, minimalist wireless earbud case, matte black finish, with subtle glowing blue LED indicator, on a dark, textured surface, studio lighting, 8k, hyperrealistic`
    *(Çeviri: Minimalist, mat siyah kaplamalı, hafif mavi yanıp sönen LED göstergeli şık bir kablosuz kulaklık kılıfının ürün fotoğrafı, koyu, dokulu bir yüzeyde, stüdyo ışıklandırması, 8k, hipergerçekçi)*
    *Bu prompt, bir ürünün detaylarını ve estetiğini vurgulayarak gerçekçi bir pazarlama görseli üretmeyi hedefler.*

*   **Fantastik Karakter (Büyülü Yaratık):**
    `full body illustration of a "Glimmerwing Sprite", a small, ethereal fae creature with iridescent butterfly wings, wearing clothes made of woven moonlight, holding a glowing staff, in an enchanted forest clearing, soft, magical lighting, style of Arthur Rackham, detailed digital painting`
    *(Çeviri: "Glimmerwing Sprite" karakterinin tam vücut illüstrasyonu, yanardöner kelebek kanatlarına sahip küçük, ruhani bir peri yaratığı, dokunmuş ay ışığından yapılmış giysiler giyiyor, parlayan bir asa tutuyor, büyülü bir orman açıklığında, yumuşak, sihirli aydınlatma, Arthur Rackham tarzında, detaylı dijital çizim)*
    *Bu prompt, fantastik bir karakterin anatomisini, kostümünü, ortamını ve sanatsal stilini belirleyerek kapsamlı bir illüstrasyon üretir.*

*   **Mekân/Çevre (Fütüristik Şehir Manzarası):**
    `wide-angle shot of a futuristic cityscape at dusk, Neo-Kyoto style, towering skyscrapers with neon signs, flying vehicles, rain-slicked streets reflecting lights, cinematic lighting, photorealistic, 4k`
    *(Çeviri: Alacakaranlıkta fütüristik bir şehir manzarasının geniş açı çekimi, Neo-Kyoto tarzı, neon tabelalı yüksek gökdelenler, uçan araçlar, ışıkları yansıtan yağmur ıslak sokaklar, sinematik aydınlatma, fotogerçekçi, 4k)*
    *Bu prompt, geniş bir sahneyi, belirli bir estetik (Neo-Kyoto), atmosfer (alacakaranlık, yağmur) ve sinematik bir kompozisyonla tanımlayarak detaylı bir çevre görseli oluşturur.*

*   **Soyut/Desen Tasarımı (Geometrik Yaprak Deseni):**
    `seamless pattern of stylized geometric leaves in emerald green and gold, on a cream background, minimalist, vector art, for textile design`
    *(Çeviri: Krem zemin üzerinde zümrüt yeşili ve altın renginde stilize edilmiş geometrik yaprakların kesintisiz deseni, minimalist, vektör sanat, tekstil tasarımı için)*
    *Bu prompt, tekrarlanabilir ve ölçeklenebilir bir desen (pattern) oluşturmayı hedefler; burada renkler, stil ve kullanım amacı (tekstil tasarımı) net bir şekilde belirtilmiştir.*

*   **Kullanıcı Arayüzü (Meditation App İkonları):**
    `vector icon set for a meditation app, featuring a lotus flower, a meditating person, and a sound wave, simple line art, calming color palette (blues, greens, purples), on a transparent background`
    *(Çeviri: Bir meditasyon uygulaması için vektörel ikon seti, lotus çiçeği, meditasyon yapan kişi ve ses dalgası içerir, basit çizgi sanatı, sakinleştirici renk paleti (mavi, yeşil, mor), şeffaf arka plan üzerinde)*
    *Bu prompt, belirli bir uygulama türü (meditasyon) için gerekli ikonları, istenen stilde (line art) ve teknik özelliklerde (transparent background) üretmeyi amaçlar.*

**(Görsel Önerisi: Bu örnek prompt'ların her biri için üretilmiş temsili görseller veya basit ikonlar. Her bir varlığın, markanın estetiğine nasıl uyduğu üzerine kısa bir açıklama eklenebilir.)**

### 1.3. Yapay Zeka Araçları: Kimler Var, Kimler Yok?

Piyasada birçok üretken yapay zeka görsel aracı bulunuyor. Her birinin kendine özgü güçlü yanları ve kullanım alanları var. Tasarım yolculuğunuzda size eşlik edecek başlıcaları tanıyalım:

*   **Midjourney:** Sanatsal, stilize, atmosferik ve genellikle yüksek estetik kalitede görseller üretmek için harikadır. Özellikle fantezi, konsept sanatı ve editoryal görseller için tercih edilir. Genellikle Discord üzerinden kullanılır. Fikirlerinizi sanatsal bir yorumla görselleştirmek istediğinizde ilk tercihiniz olabilir.
*   **Adobe Firefly:** Ticari kullanıma uygun, etik kaynaklardan eğitilmiş olmasıyla öne çıkar. Adobe Creative Cloud uygulamalarıyla (Photoshop, Illustrator) kusursuz entegrasyonu sayesinde tasarım akışına doğrudan dahil edilebilir. Gerçekçi görseller, ürün mock-up'ları ve metinden vektör üretimi gibi alanlarda güçlüdür. Tasarım projelerinizde kullanıma hazır, güvenilir görseller arıyorsanız idealdir.
*   **DALL-E 3 (ChatGPT Plus / Copilot ile erişilebilir):** Prompt'ları çok doğru anlama ve metin-görsel tutarlılığında oldukça başarılıdır. Yaratıcı, detaylı ve bazen de eğlenceli sonuçlar elde etmek için idealdir. Özellikle belirli kompozisyonları, karakterleri ve metinleri görselleştirmede etkilidir. Fikrinizi tam olarak tarif edip, onu birebir görselleştirmesini istediğinizde kullanışlıdır.

## Uygulama: Konsept Geliştirme

Bu bölümde, öğrendiğimiz teorik bilgileri ve prompt yazma tekniklerini pratik bir projede uygulayacağız. Amacımız, yapay zeka ile fikirlerimizi somut görsellere dönüştürmek ve markaların görsel kimliklerini oluşturmaktır.

### 2.1. Projemiz: "Artisan Roast" Kahve Markası - Bir Vaka Analizi

Projemiz, sürdürülebilir ve organik kahve çekirdekleri sunan, doğallık, zanaatkarlık ve modern sadeliği birleştiren "Artisan Roast" adında hayali bir kahve markası için görsel bir kimlik oluşturmak. Hedef kitlemiz, kaliteye önem veren, bilinçli tüketicilerdir. Bu brief'i göz önünde bulundurarak, yapay zeka araçlarıyla markanın ruhunu yansıtan ilk görsellerimizi üreteceğiz. Proje brief'ini detaylıca incelemek için `AI_Tasarim_Dersi/Proje_Briefi.md` dosyasını referans alacağız. Bu vaka analizi, soyut bir konseptin nasıl somut bir görsel kimliğe dönüştürülebileceğini gösterecektir.

**(Görsel Önerisi: Markanın felsefesini yansıtan birkaç anahtar görsel (örn: yeşil kahve yaprakları, rustik ahşap dokular, sade seramik kupalar, el değirmeni gibi zanaatkarlık öğeleri). Bu görseller, projenin genel estetik yönelimini ilk bakışta anlamayı sağlar.)**

### 2.2. Görev 1: Markanın Görsel Ruhu - Mood Board Oluşturma Süreci

Bir projenin görsel dilini oluşturmanın ilk ve en kritik adımlarından biri, ilham kaynaklarımızı toplamak ve organize etmektir. Mood board'lar, projenin genel estetiğini, renk paletini, dokularını, stilini ve atmosferini belirlememize yardımcı olur. Bu görevde, "Artisan Roast" markasının kimliğini en iyi yansıtabilecek görselleri yapay zeka ile üreterek kendi ilham panomuzu oluşturacağız. Bu süreç, yaratıcı bir beyin fırtınası ve kavramsal geliştirmenin birleşimidir.

*   **Amaç:** Markanın doğallığını, zanaatkarlığını ve modern sadeliğini yansıtan soyut dokular, renk paletleri, olası kompozisyonlar ve genel atmosferi yakalamak. Bu, markanın hedef kitlesiyle kuracağı görsel bağı şekillendirecektir.
*   **Ne Üreteceğiz?** Kahve çekirdeklerinin yakın çekimleri, yeşil kahve yaprakları, doğal ahşap dokuları, markanın kullanabileceği renk paletlerini temsil eden görseller, minimalist ambalaj tasarımları için ilham veren kompozisyonlar. Bu görseller, projenin geri kalanı için bir görsel referans noktası ve paydaşlarla iletişimde bir araç görevi görecektir.
*   **Örnek Prompt:** `mood board for an organic coffee brand, earthy tones, minimalist, with textures of coffee beans and green leaves, artisanal feel, natural light`

**(Görsel Önerisi: Örnek bir mood board tasarımı. Farklı doku, renk ve stil örneklerini bir araya getiren, ilham verici bir kolaj. Bu kolaj, projenin başlangıç estetiğini somutlaştırır.)**

### 2.3. Canlı Uygulama: Prompt'larla Deneysel Üretim ve İyileştirme

Şimdi, öğrendiğimiz prompt yazma tekniklerini kullanarak "Artisan Roast" markası için canlı olarak görseller üreteceğiz. Öğrencilerden gelen sorular ve öneriler doğrultusunda prompt'larımızı adım adım iyileştirecek, farklı AI araçlarının çıktılarını karşılaştıracağız. Hangi kelimelerin, hangi stillerin ve hangi parametrelerin daha etkili sonuçlar verdiğini bu uygulamalı bölümde göreceğiz. Bu, AI ile etkileşim kurarken iteratif bir yaklaşımın önemini vurgulayacaktır. Lütfen aklınızdaki fikirleri, denemek istediğiniz konseptleri veya karşılaştığınız zorlukları benimle paylaşmaktan çekinmeyin! Hep birlikte en iyi sonuçları elde etmeye çalışacağız.

### 2.4. Görev 2: Markamız İçin Temel Yapı Taşları - Varlık Üretimi

Mood board'umuz markanın genel atmosferini belirledikten sonra, şimdi bu atmosferi somutlaştıracak spesifik görsel varlıklar (assets) üretme zamanı. Bu varlıklar, ilerleyen haftalarda tasarım projelerimizde doğrudan kullanabileceğimiz, markanın diline uygun hazır elementler olacaktır. Bu görev, yapay zekanın sadece konsept görselleri değil, aynı zamanda tasarımın temel yapı taşlarını da üretebildiğini gösterecektir.

*   **Amaç:** Markanın kimliğini yansıtan, tasarım projelerinde kullanılmaya hazır ikonlar, illüstrasyonlar, desenler veya logo taslakları oluşturmak.
*   **Ne Üreteceğiz?**
    *   **İkonlar:** Örneğin, markanın logosu olabilecek minimalist bir kahve fincanı, kahve çekirdeği veya kahve yaprağı ikonu. Bu ikonlar, web sitelerinde, uygulamalarda veya baskılı materyallerde kullanılabilir. Ölçeklenebilir vektörel formatta üretilirse daha fonksiyonel olacaktır.
    *   **İllüstrasyonlar:** Markanın hikayesini anlatan stilize görseller (örn: bir kahve çiftçisi, kahve bitkisinin büyüme aşamaları, keyifli bir kahve anı).
    *   **Desenler (Patterns):** Kahve çekirdekleri, yapraklar veya geometrik formlardan oluşan, markanın renk paletini kullanan ve ambalajlarda veya web sitesi arka planlarında kullanılabilecek tekrarlayan grafikler. Bu desenler, marka tutarlılığı sağlamak için idealdir.
*   **Örnek Prompt (İkon):** `vector icon of a coffee cup, minimalist, flat design, single line, on a white background`
*   **Örnek Prompt (İllüstrasyon):** `digital illustration of a coffee plant with red berries, detailed, botanical art style, in a slightly desaturated color palette`

**(Görsel Önerisi: Yan yana birkaç farklı üretilmiş varlık örneği (örn: bir ikon seti, stilize bir illüstrasyon, tekrarlayan bir desen). Her bir varlığın, markanın estetiğine nasıl uyduğu üzerine kısa bir açıklama eklenebilir.)**

## Değerlendirme ve Kapanış

### 3.1. Üretkenlik Değerlendirmesi ve Geri Bildirim Mekanizmaları

Bu bölümde, öğrencilerimizin ürettiği ilham panolarını ve görsel varlıkları birlikte inceleyeceğiz. Her öğrenci, kullandığı prompt'ları, karşılaştığı zorlukları, denediği yaklaşımları ve edindiği çıktıları kısaca paylaşacak. Öğretmen ve diğer öğrenciler, yapılan çalışmaları yapıcı geri bildirimlerle değerlendirecek. Hangi yaklaşımların daha başarılı olduğunu, hangi prompt'ların neden daha iyi sonuçlar verdiğini tartışarak hep birlikte öğreneceğiz. Bu, hem bireysel gelişimi teşvik edecek hem de sınıf içi bilgi paylaşımını ve farklı bakış açılarını zenginleştirecektir. Bu değerlendirme süreci, AI ile görsel üretimi konusundaki pratik becerileri pekiştirmeyi amaçlar.

### 3.2. Haftanın Özeti ve Gelecek Haftaya Bakış

*   **Bu Hafta:** Yapay zekanın tasarım dünyasındaki yerini keşfettik, etkili prompt yazmanın temellerini öğrendik ve "Artisan Roast" markası için ilk fikirlerimizi ve temel görsel varlıklarımızı ürettik. Artık AI ile "konuşmayı" biliyor ve hayallerimizi görsellere dökebiliyoruz! Yapay zekanın tasarım sürecine entegrasyonunun ilk adımlarını attık.
*   **Ödev:** Lütfen bu hafta ürettiğiniz en beğendiğiniz 5-10 görseli (mood board'lar ve varlıklar) kaydedin ve bir sonraki derse hazırlanın. Neden bu görselleri seçtiğinizi düşünün; hangi özelliklerinin hoşunuza gittiğini not alın. Bu, ileriki aşamada hangi estetik yönelimleri takip edeceğiniz konusunda size rehberlik edecektir.
*   **Gelecek Hafta:** Bu hafta ürettiğimiz ham AI görsellerini, tasarım dünyasının vazgeçilmez programları olan Adobe Photoshop ve Illustrator'a taşıyacağız. Yapay zeka ile üretilen bu ham materyalleri, kusursuz ve profesyonel tasarımlara nasıl dönüştüreceğimizi öğreneceğiz. AI'nın bir araç olarak tasarım sürecine nasıl entegre edildiğini adım adım göreceğiz. Yeni haftada görüşmek üzere!

**(Görsel Önerisi: Bir yanda bu hafta üretilmiş ham, stilize edilmemiş AI görsellerinin bir kolajı; diğer yanda ise bu görsellerin önümüzdeki hafta işlenerek oluşturulmuş, bitmiş bir sosyal medya gönderisi veya afiş tasarımı (bir "teaser" efektiyle). Bu, öğrencilere gelecek dersin potansiyelini ve öğrenme hedefini görsel olarak sunacaktır.)**
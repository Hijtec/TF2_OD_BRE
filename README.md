Jak zprovoznit stack (Windows 10)
##################################
I. Instalace stacku TF2

-Nejjednodušší metoda (Conda)
$conda env create -f environment.yml (kde envirnoment.yml je cesta k souboru ROOT slozky)

-nebo přes pip, nutno zajistit použití Python 3.8.x
$pip install -r requirements.txt (kde requirements.txt je cesta k souboru v ROOT slozky)
##################################
II. Aktivace prostředí
-přes příkazový řádek
$conda activate BRE-TF-2_3-p38

pokud bylo instalováno přes pip není potřeba aktivace prostředí
##################################
III. Nastavení aplikace
1) otevřít src/main/flags_global.py
2) nastavit PATHS (absolutní cesty ve tvaru r"CESTAKSOUBORU/SLOZCE") přes windows explorera jednoduše shift+pravý klik -> copy as path
je třeba změnit následující cesty:

detector_elements_model_path
label_map_path_detection
classification_floor_button_model_path
label_map_path_button_classification
output_image_save_detection
output_image_save_classification

3) nastavení obrazového vstupu
image_input_mode řeší vstup do algoritmu, možnosti nastavení:

camera - čtení nejnovějšího snímku připojené kamery s indexem dle camera_device_used (asynchronní)
folder - rekurzivně čte obrázky ze složky definované v image_input_folder_path
video_per_frame - čte z videosouboru po jednotlivých snímcích (synchronní)
video_async - spustí video v pozadí a čte snímky když je požadován nový vstup (asynchronní)

##################################
IV. Spouštění aplikace
Aplikace je nyní nastavena a je spouštěna přes modul src/main/App.py

Conda prikazak:
(BRE-TF-2_3-p38) "musi byt aktivni toto prostredi"
$python -m PATH_TO_APP_PY

Prikazak bezny (ujistit se ze volam python verze 3.8)
$python -m PATH_TO_APP_PY

IDE:
v konfiguraci zvolit spravny environment a spustit/debugovat App.py

##################################
V. Možné problémy
Mozna bude nutno pridat PATH vytvoreneho conda environmentu do PATH v environmental variables ve Windowsech
Je potreba se ujistit, ze verze tensorflow a numpy je presne stejna jako uvedeno v requirements.txt (proto doporucuji Condu)



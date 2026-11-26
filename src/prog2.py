PROGRAMME 2 — CALIBRATION DES ROI ET DES VALEURS DE RÉFÉRENCE

Fonctions :
1. Sélection manuelle des ROI sur une image de référence de chaque face.
2. Sauvegarde des ROI en pixels et en coordonnées normalisées.
3. Application des mêmes ROI aux 10 images conformes de chaque face.
4. Mesure du contraste, de la luminosité, de la saturation, des contours
   et de la texture dans chaque ROI.
5. Création d'un CSV complet et d'un JSON de références.
6. Calcul de seuils OK/NOK seulement lorsque des images NOK seront ajoutées.

IMPORTANT :
La base actuelle contient uniquement des images conformes. Le programme calcule donc
pour le moment des plages de référence des pièces OK, et non des seuils définitifs
séparant les pièces OK des pièces NOK.
"""

import cv2
import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 1. CHEMINS

DOSSIER_BASE = Path(
    r"C:\Users\zaher\OneDrive\Desktop\Base_Donnees_Tambours_MTV9"
)

# Images conformes issues du Programme 1.
DOSSIER_OK_AVANT = DOSSIER_BASE / "Face_Avant_Centree"
DOSSIER_OK_ARRIERE = DOSSIER_BASE / "Face_Arriere_Centree"

# Emplacements prévus pour les futures images NOK.
DOSSIER_NOK_AVANT = DOSSIER_BASE / "Face_Avant_NOK_Centree"
DOSSIER_NOK_ARRIERE = DOSSIER_BASE / "Face_Arriere_NOK_Centree"

DOSSIER_RESULTATS = DOSSIER_BASE / "Calibration_ROI_Seuils"
DOSSIER_IMAGES_ROI = DOSSIER_RESULTATS / "Images_ROI"
DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)
DOSSIER_IMAGES_ROI.mkdir(parents=True, exist_ok=True)

FICHIER_ROI = DOSSIER_RESULTATS / "rois_calibration.json"
FICHIER_MESURES = DOSSIER_RESULTATS / "mesures_calibration.csv"
FICHIER_REFERENCES = DOSSIER_RESULTATS / "references_et_seuils.json"


# 2. PARAMÈTRES

EXTENSIONS_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LARGEUR_APERCU = 1100

# Paramètres provisoires utilisés pour les mesures.
SEUIL_SATURATION_PROVISOIRE = 45
SEUIL_LUMINOSITE_MIN_PROVISOIRE = 35
SEUIL_LAPLACIEN_PROVISOIRE = 45
FACTEUR_MAD = 3.5

# ROI retenues pour les essais.
NOMS_ROI_FACE_AVANT = [
    "piece_complete",
    "zone_centrale_roulement_circlips",
    "zone_cinq_trous",
    "zone_marquage",
    "zone_protection_peinte",
    "zone_interdite_peinte",
    "surface_generale",
]

NOMS_ROI_FACE_ARRIERE = [
    "piece_complete",
    "zone_centrale_roulement",
    "zone_cinq_trous",
    "zone_marquage_peripherique",
    "zone_protection_peinte",
    "zone_interdite_peinte",
    "surface_generale",
]

# 3. LECTURE / SAUVEGARDE AVEC ACCENTS WINDOWS

def lire_image_unicode(chemin_image):
    """Lit une image même si son chemin contient des accents."""
    donnees = np.fromfile(str(chemin_image), dtype=np.uint8)
    image = cv2.imdecode(donnees, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Impossible de lire : {chemin_image}")
    return image


def sauvegarder_image_unicode(chemin_image, image):
    """Sauvegarde une image avec un chemin pouvant contenir des accents."""
    extension = chemin_image.suffix.lower()
    if extension not in EXTENSIONS_IMAGES:
        extension = ".png"
    succes, donnees = cv2.imencode(extension, image)
    if not succes:
        raise IOError(f"Impossible d'encoder : {chemin_image}")
    donnees.tofile(str(chemin_image))


def lister_images(dossier, obligatoire=True):
    """Retourne toutes les images d'un dossier dans l'ordre alphabétique."""
    if not dossier.exists():
        if obligatoire:
            raise FileNotFoundError(f"Dossier introuvable : {dossier}")
        return []

    images = [
        chemin for chemin in dossier.iterdir()
        if chemin.is_file() and chemin.suffix.lower() in EXTENSIONS_IMAGES
    ]
    return sorted(images)


# 4. AFFICHAGE ET SÉLECTION DES ROI

def creer_apercu(image, largeur_max=LARGEUR_APERCU):
    """Réduit seulement l'affichage pour qu'il tienne sur l'écran."""
    hauteur, largeur = image.shape[:2]
    if largeur <= largeur_max:
        return image.copy(), 1.0

    facteur = largeur_max / float(largeur)
    nouvelle_hauteur = int(round(hauteur * facteur))
    apercu = cv2.resize(
        image,
        (largeur_max, nouvelle_hauteur),
        interpolation=cv2.INTER_AREA,
    )
    return apercu, facteur


def selectionner_rois(image, noms_rois, nom_face):
    """
    Sélectionne les ROI une par une.

    Pour chaque ROI :
    - tracer le rectangle avec la souris ;
    - valider avec Entrée ou Espace ;
    - appuyer sur C ou Échap pour ignorer la ROI.
    """
    rois = {}
    apercu_base, facteur = creer_apercu(image)

    for nom_roi in noms_rois:
        apercu = apercu_base.copy()
        cv2.putText(
            apercu,
            f"Selectionner : {nom_roi}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

        fenetre = f"{nom_face} - {nom_roi}"
        x, y, largeur, hauteur = cv2.selectROI(
            fenetre,
            apercu,
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyWindow(fenetre)

        if largeur == 0 or hauteur == 0:
            print(f"[ROI ignorée] {nom_roi}")
            continue

        roi_originale = (
            int(round(x / facteur)),
            int(round(y / facteur)),
            int(round(largeur / facteur)),
            int(round(hauteur / facteur)),
        )
        rois[nom_roi] = roi_originale
        print(f"[ROI] {nom_face}/{nom_roi} = {roi_originale}")

    return rois


def normaliser_roi(roi, largeur_image, hauteur_image):
    """Convertit une ROI pixel en coordonnées relatives comprises entre 0 et 1."""
    x, y, largeur, hauteur = roi
    return {
        "x_norm": float(x / largeur_image),
        "y_norm": float(y / hauteur_image),
        "largeur_norm": float(largeur / largeur_image),
        "hauteur_norm": float(hauteur / hauteur_image),
    }


def denormaliser_roi(roi_normalisee, largeur_image, hauteur_image):
    """Convertit une ROI relative en pixels pour l'image actuelle."""
    return (
        int(round(roi_normalisee["x_norm"] * largeur_image)),
        int(round(roi_normalisee["y_norm"] * hauteur_image)),
        int(round(roi_normalisee["largeur_norm"] * largeur_image)),
        int(round(roi_normalisee["hauteur_norm"] * hauteur_image)),
    )


def dessiner_rois(image, rois, couleur):
    """Dessine les ROI sur une copie de l'image."""
    sortie = image.copy()
    for nom_roi, (x, y, largeur, hauteur) in rois.items():
        cv2.rectangle(
            sortie,
            (x, y),
            (x + largeur, y + hauteur),
            couleur,
            4,
        )
        cv2.putText(
            sortie,
            nom_roi,
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            couleur,
            2,
        )
    return sortie


def creer_fichier_rois():
    """Sélectionne les ROI sur la première image OK de chaque face."""
    images_avant = lister_images(DOSSIER_OK_AVANT)
    images_arriere = lister_images(DOSSIER_OK_ARRIERE)

    if not images_avant or not images_arriere:
        raise RuntimeError("Les dossiers centrés doivent contenir des images.")

    image_avant = lire_image_unicode(images_avant[0])
    image_arriere = lire_image_unicode(images_arriere[0])

    print("\n=== ROI FACE AVANT ===")
    rois_avant = selectionner_rois(
        image_avant,
        NOMS_ROI_FACE_AVANT,
        "Face_Avant",
    )

    print("\n=== ROI FACE ARRIÈRE ===")
    rois_arriere = selectionner_rois(
        image_arriere,
        NOMS_ROI_FACE_ARRIERE,
        "Face_Arriere",
    )

    h_av, w_av = image_avant.shape[:2]
    h_ar, w_ar = image_arriere.shape[:2]

    donnees = {
        "date_creation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "avertissement": (
            "ROI définies sur images iPhone centrées. "
            "Recalibration obligatoire avec les caméras industrielles finales."
        ),
        "image_reference_face_avant": images_avant[0].name,
        "image_reference_face_arriere": images_arriere[0].name,
        "face_avant": {
            "pixels": {nom: list(roi) for nom, roi in rois_avant.items()},
            "normalisees": {
                nom: normaliser_roi(roi, w_av, h_av)
                for nom, roi in rois_avant.items()
            },
        },
        "face_arriere": {
            "pixels": {nom: list(roi) for nom, roi in rois_arriere.items()},
            "normalisees": {
                nom: normaliser_roi(roi, w_ar, h_ar)
                for nom, roi in rois_arriere.items()
            },
        },
    }

    with open(FICHIER_ROI, "w", encoding="utf-8") as fichier:
        json.dump(donnees, fichier, indent=4, ensure_ascii=False)

    sauvegarder_image_unicode(
        DOSSIER_IMAGES_ROI / "reference_face_avant_ROI.png",
        dessiner_rois(image_avant, rois_avant, (0, 255, 0)),
    )
    sauvegarder_image_unicode(
        DOSSIER_IMAGES_ROI / "reference_face_arriere_ROI.png",
        dessiner_rois(image_arriere, rois_arriere, (255, 0, 0)),
    )

    print(f"\nROI sauvegardées : {FICHIER_ROI}")
    return donnees


def charger_rois():
    """Charge le fichier JSON des ROI."""
    if not FICHIER_ROI.exists():
        raise FileNotFoundError(
            "Le fichier ROI n'existe pas. Lance d'abord le mode 1 ou 3."
        )
    with open(FICHIER_ROI, "r", encoding="utf-8") as fichier:
        return json.load(fichier)


# 5. EXTRACTION ET MESURES DANS UNE ROI

def extraire_roi(image, roi):
    """Extrait une ROI en sécurisant ses limites."""
    x, y, largeur, hauteur = roi
    h_img, w_img = image.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(w_img, int(x + largeur))
    y2 = min(h_img, int(y + hauteur))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"ROI invalide : {roi}")

    return image[y1:y2, x1:x2]


def calculer_caracteristiques_roi(roi_bgr):
    """Calcule les valeurs qui serviront ensuite à choisir les seuils."""
    gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    _, saturation, luminosite = cv2.split(hsv)

    gris_p05 = float(np.percentile(gris, 5))
    gris_p95 = float(np.percentile(gris, 95))

    masque_couleur = np.zeros_like(saturation, dtype=np.uint8)
    masque_couleur[
        (saturation > SEUIL_SATURATION_PROVISOIRE)
        & (luminosite > SEUIL_LUMINOSITE_MIN_PROVISOIRE)
    ] = 255

    gris_lisse = cv2.GaussianBlur(gris, (5, 5), 0)
    canny = cv2.Canny(gris_lisse, 50, 150)

    laplacien = cv2.Laplacian(gris, cv2.CV_64F)
    lap_abs = np.uint8(np.clip(np.abs(laplacien), 0, 255))
    _, masque_texture = cv2.threshold(
        lap_abs,
        SEUIL_LAPLACIEN_PROVISOIRE,
        255,
        cv2.THRESH_BINARY,
    )

    return {
        "gris_moyenne": float(np.mean(gris)),
        "gris_ecart_type": float(np.std(gris)),
        "gris_mediane": float(np.median(gris)),
        "gris_p05": gris_p05,
        "gris_p95": gris_p95,
        "contraste_p90": float(gris_p95 - gris_p05),
        "saturation_moyenne": float(np.mean(saturation)),
        "saturation_mediane": float(np.median(saturation)),
        "luminosite_moyenne": float(np.mean(luminosite)),
        "ratio_pixels_colores": float(np.mean(masque_couleur > 0)),
        "densite_contours": float(np.mean(canny > 0)),
        "nombre_pixels_contours": int(np.sum(canny > 0)),
        "variance_laplacien": float(np.var(laplacien)),
        "ratio_texture": float(np.mean(masque_texture > 0)),
        "ratio_pixels_sombres": float(np.mean(gris < 40)),
        "ratio_pixels_clairs": float(np.mean(gris > 215)),
    }


def analyser_image(chemin_image, label, face, rois_normalisees):
    """Applique toutes les ROI à une image et retourne les mesures."""
    image = lire_image_unicode(chemin_image)
    hauteur, largeur = image.shape[:2]
    lignes = []
    rois_pixels = {}

    for nom_roi, roi_norm in rois_normalisees.items():
        roi_pixels = denormaliser_roi(roi_norm, largeur, hauteur)
        rois_pixels[nom_roi] = roi_pixels
        zone = extraire_roi(image, roi_pixels)
        mesures = calculer_caracteristiques_roi(zone)

        ligne = {
            "image": chemin_image.name,
            "label": label,
            "face": face,
            "roi": nom_roi,
            "largeur_image": largeur,
            "hauteur_image": hauteur,
            "roi_x": roi_pixels[0],
            "roi_y": roi_pixels[1],
            "roi_largeur": roi_pixels[2],
            "roi_hauteur": roi_pixels[3],
        }
        ligne.update(mesures)
        lignes.append(ligne)

    couleur = (0, 255, 0) if face == "face_avant" else (255, 0, 0)
    image_annotee = dessiner_rois(image, rois_pixels, couleur)
    sauvegarder_image_unicode(
        DOSSIER_IMAGES_ROI / f"{label}_{face}_{chemin_image.stem}_ROI.png",
        image_annotee,
    )

    return lignes

# 6. ANALYSE COMPLÈTE DE LA BASE

def analyser_base_images():
    """Analyse les images OK et les images NOK si elles existent plus tard."""
    donnees_roi = charger_rois()
    rois_avant = donnees_roi["face_avant"]["normalisees"]
    rois_arriere = donnees_roi["face_arriere"]["normalisees"]

    sources = [
        ("OK", "face_avant", DOSSIER_OK_AVANT, rois_avant, True),
        ("OK", "face_arriere", DOSSIER_OK_ARRIERE, rois_arriere, True),
        ("NOK", "face_avant", DOSSIER_NOK_AVANT, rois_avant, False),
        ("NOK", "face_arriere", DOSSIER_NOK_ARRIERE, rois_arriere, False),
    ]

    toutes_mesures = []

    for label, face, dossier, rois, obligatoire in sources:
        images = lister_images(dossier, obligatoire)
        print(f"\n=== {label} / {face} : {len(images)} image(s) ===")

        for numero, chemin in enumerate(images, start=1):
            print(f"[{numero}/{len(images)}] {chemin.name}")
            try:
                toutes_mesures.extend(
                    analyser_image(chemin, label, face, rois)
                )
            except Exception as erreur:
                print(f"[ERREUR] {chemin.name} : {erreur}")

    sauvegarder_mesures_csv(toutes_mesures)
    calculer_references_et_seuils(toutes_mesures)
    return toutes_mesures


def sauvegarder_mesures_csv(mesures):
    """Crée le tableau CSV contenant une ligne par image et par ROI."""
    if not mesures:
        print("Aucune mesure à sauvegarder.")
        return

    debut = [
        "image", "label", "face", "roi",
        "largeur_image", "hauteur_image",
        "roi_x", "roi_y", "roi_largeur", "roi_hauteur",
    ]
    toutes_colonnes = sorted(set().union(*(ligne.keys() for ligne in mesures)))
    colonnes = debut + [c for c in toutes_colonnes if c not in debut]

    with open(
        FICHIER_MESURES,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as fichier:
        writer = csv.DictWriter(fichier, fieldnames=colonnes, delimiter=";")
        writer.writeheader()
        writer.writerows(mesures)

    print(f"\nMesures sauvegardées : {FICHIER_MESURES}")

# 7. STATISTIQUES ET SEUILS

def calculer_statistiques(valeurs):
    """Calcule une plage robuste de référence pour les pièces conformes."""
    tableau = np.asarray(valeurs, dtype=np.float64)
    if tableau.size == 0:
        return None

    mediane = float(np.median(tableau))
    mad = float(np.median(np.abs(tableau - mediane)))
    sigma_robuste = 1.4826 * mad
    marge_min = max(1e-6, 0.02 * max(abs(mediane), 1.0))
    demi_largeur = max(FACTEUR_MAD * sigma_robuste, marge_min)

    p05 = float(np.percentile(tableau, 5))
    p95 = float(np.percentile(tableau, 95))
    borne_basse = min(mediane - demi_largeur, p05)
    borne_haute = max(mediane + demi_largeur, p95)

    return {
        "nombre_images": int(tableau.size),
        "minimum": float(np.min(tableau)),
        "maximum": float(np.max(tableau)),
        "moyenne": float(np.mean(tableau)),
        "ecart_type": float(np.std(tableau)),
        "mediane": mediane,
        "percentile_05": p05,
        "percentile_95": p95,
        "mad": mad,
        "plage_reference_OK_provisoire": {
            "borne_basse": float(borne_basse),
            "borne_haute": float(borne_haute),
        },
    }


def trouver_seuil_supervise(valeurs_ok, valeurs_nok):
    """Calcule un seuil simple quand les deux groupes OK et NOK existent."""
    ok = np.asarray(valeurs_ok, dtype=np.float64)
    nok = np.asarray(valeurs_nok, dtype=np.float64)
    if ok.size == 0 or nok.size == 0:
        return None

    med_ok = float(np.median(ok))
    med_nok = float(np.median(nok))

    if med_ok > med_nok:
        limite_ok = float(np.percentile(ok, 5))
        limite_nok = float(np.percentile(nok, 95))
        seuil = (limite_ok + limite_nok) / 2.0
        regle = "OK_si_valeur_superieure_ou_egale"
        separation = limite_ok > limite_nok
    else:
        limite_ok = float(np.percentile(ok, 95))
        limite_nok = float(np.percentile(nok, 5))
        seuil = (limite_ok + limite_nok) / 2.0
        regle = "OK_si_valeur_inferieure_ou_egale"
        separation = limite_ok < limite_nok

    return {
        "seuil_suggere": float(seuil),
        "regle": regle,
        "groupes_clairement_separes": bool(separation),
        "mediane_OK": med_ok,
        "mediane_NOK": med_nok,
    }


def calculer_references_et_seuils(mesures):
    """Crée le JSON final de références et de seuils."""
    caracteristiques = [
        "gris_moyenne",
        "gris_ecart_type",
        "gris_mediane",
        "gris_p05",
        "gris_p95",
        "contraste_p90",
        "saturation_moyenne",
        "saturation_mediane",
        "luminosite_moyenne",
        "ratio_pixels_colores",
        "densite_contours",
        "nombre_pixels_contours",
        "variance_laplacien",
        "ratio_texture",
        "ratio_pixels_sombres",
        "ratio_pixels_clairs",
    ]

    contient_nok = any(ligne["label"] == "NOK" for ligne in mesures)
    resultat = {
        "date_creation": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "contient_images_NOK": contient_nok,
        "avertissement": (
            "Sans images NOK, les bornes sont des références provisoires des pièces OK, "
            "pas des seuils industriels définitifs."
        ),
        "references": {},
    }

    combinaisons = sorted({(m["face"], m["roi"]) for m in mesures})

    for face, roi in combinaisons:
        cle = f"{face}/{roi}"
        resultat["references"][cle] = {}

        for caracteristique in caracteristiques:
            valeurs_ok = [
                float(m[caracteristique]) for m in mesures
                if m["face"] == face
                and m["roi"] == roi
                and m["label"] == "OK"
            ]
            valeurs_nok = [
                float(m[caracteristique]) for m in mesures
                if m["face"] == face
                and m["roi"] == roi
                and m["label"] == "NOK"
            ]

            resultat["references"][cle][caracteristique] = {
                "statistiques_OK": calculer_statistiques(valeurs_ok),
                "statistiques_NOK": calculer_statistiques(valeurs_nok),
                "seuil_OK_NOK_suggere": trouver_seuil_supervise(
                    valeurs_ok,
                    valeurs_nok,
                ),
            }

    with open(FICHIER_REFERENCES, "w", encoding="utf-8") as fichier:
        json.dump(resultat, fichier, indent=4, ensure_ascii=False)

    print(f"Références sauvegardées : {FICHIER_REFERENCES}")

    if not contient_nok:
        print("\nAucune image NOK trouvée.")
        print("Les plages calculées décrivent uniquement les 20 images conformes.")
        print("Ajoute plus tard des images NOK pour obtenir de vrais seuils OK/NOK.")


# 8. MENU PRINCIPAL

def main():
    """Affiche le menu du Programme 2."""
    print("============================================================")
    print(" PROGRAMME 2 — CALIBRATION ROI ET SEUILS")
    print("============================================================")
    print(f"Images OK avant   : {DOSSIER_OK_AVANT}")
    print(f"Images OK arrière : {DOSSIER_OK_ARRIERE}")
    print(f"Résultats         : {DOSSIER_RESULTATS}")

    print("\n1 - Sélectionner les ROI")
    print("2 - Analyser la base avec les ROI existantes")
    print("3 - Sélectionner les ROI puis analyser la base")
    print("q - Quitter")

    choix = input("\nTon choix : ").strip().lower()

    try:
        if choix == "1":
            creer_fichier_rois()
        elif choix == "2":
            analyser_base_images()
        elif choix == "3":
            creer_fichier_rois()
            analyser_base_images()
        elif choix == "q":
            print("Fin du programme.")
        else:
            print("Choix invalide.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

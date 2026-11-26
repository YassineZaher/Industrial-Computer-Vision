PROGRAMME 3 — INSPECTION FINALE DES TAMBOURS MTV9 À DEUX CAMÉRAS

        - acquiert la face avant et la face arrière ;
        - applique le même prétraitement que le Programme 1 ;
        - charge automatiquement les ROI du Programme 2 ;
        - calcule les caractéristiques dans chaque ROI ;
        - compare les valeurs aux références des pièces conformes ;
        - produit une décision finale ;
        - enregistre les résultats dans un CSV et un fichier JSON ;
        - sauvegarde les images centrées et annotées.

Lorsque des images NOK réelles seront disponibles, il faudra relancer le Programme 2
afin de calculer des seuils supervisés plus fiables.
"""

# IMPORTATIONS


import cv2
import csv
import json
import time
import numpy as np

from pathlib import Path
from datetime import datetime


# 1. CHEMINS DU PROJET

DOSSIER_BASE = Path(
    r"C:\Users\zaher\OneDrive\Desktop\Base_Donnees_Tambours_MTV9"
)

DOSSIER_CALIBRATION = DOSSIER_BASE / "Calibration_ROI_Seuils"
FICHIER_ROI = DOSSIER_CALIBRATION / "rois_calibration.json"
FICHIER_REFERENCES = DOSSIER_CALIBRATION / "references_et_seuils.json"

DOSSIER_INSPECTION = DOSSIER_BASE / "Inspection_Finale"
DOSSIER_IMAGES_BRUTES = DOSSIER_INSPECTION / "Images_Brutes"
DOSSIER_IMAGES_CENTREES = DOSSIER_INSPECTION / "Images_Centrees"
DOSSIER_IMAGES_ANNOTEES = DOSSIER_INSPECTION / "Images_Annotees"
DOSSIER_DETAILS_JSON = DOSSIER_INSPECTION / "Details_JSON"

FICHIER_RESULTATS_CSV = DOSSIER_INSPECTION / "resultats_inspection.csv"

for dossier in (
    DOSSIER_INSPECTION,
    DOSSIER_IMAGES_BRUTES,
    DOSSIER_IMAGES_CENTREES,
    DOSSIER_IMAGES_ANNOTEES,
    DOSSIER_DETAILS_JSON
):
    dossier.mkdir(parents=True, exist_ok=True)


# 2. CONFIGURATION DU MODE D'EXÉCUTION

MODE_TEST_IMAGES = True

PATH_TEST_FACE_AVANT = (
    DOSSIER_BASE
    / "Face_Avant_Centree"
    / "AV_OK_0001_CENTREE.png"
)

PATH_TEST_FACE_ARRIERE = (
    DOSSIER_BASE
    / "Face_Arriere_Centree"
    / "AR_OK_0001_CENTREE.png"
)

CAMERA_HAUT_INDEX = 0
CAMERA_BAS_INDEX = 1

LARGEUR_CAMERA = 4096
HAUTEUR_CAMERA = 3000

NOMBRE_IMAGES_A_JETER = 5

TAILLE_IMAGE_NORMALISEE = 3000

MARGE_AUTOUR_PIECE = 0.06

AFFICHER_IMAGES = True

SAUVEGARDER_IMAGES = True

VALIDATION_DETECTION_INTERACTIVE = False

EXTENSIONS_IMAGES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


# 3. PARAMÈTRES DE DÉCISION

NOMBRE_ANOMALIES_TOLEREES = 0

MARGE_SECURITE_REFERENCES = 0.10

UTILISER_STATUT_A_VERIFIER_SANS_NOK = True

CARACTERISTIQUES_DECISION = {
    "zone_centrale_roulement_circlips": [
        "gris_ecart_type",
        "densite_contours",
        "variance_laplacien",
        "ratio_pixels_sombres"
    ],
    "zone_centrale_roulement": [
        "gris_ecart_type",
        "densite_contours",
        "variance_laplacien",
        "ratio_pixels_sombres"
    ],
    "zone_cinq_trous": [
        "densite_contours",
        "nombre_pixels_contours",
        "ratio_pixels_sombres"
    ],
    "zone_marquage": [
        "gris_ecart_type",
        "contraste_p90",
        "densite_contours"
    ],
    "zone_marquage_peripherique": [
        "gris_ecart_type",
        "contraste_p90",
        "densite_contours"
    ],
    "zone_protection_peinte": [
        "saturation_moyenne",
        "ratio_pixels_colores",
        "luminosite_moyenne"
    ],
    "zone_interdite_peinte": [
        "saturation_moyenne",
        "ratio_pixels_colores",
        "luminosite_moyenne"
    ],
    "surface_generale": [
        "ratio_texture",
        "variance_laplacien",
        "densite_contours"
    ]
}


# 4. LECTURE ET SAUVEGARDE DES IMAGES

def lire_image_unicode(chemin_image):
    """Lit une image même si le chemin Windows contient des accents."""

    donnees = np.fromfile(str(chemin_image), dtype=np.uint8)
    image = cv2.imdecode(donnees, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Impossible de lire l'image : {chemin_image}"
        )

    return image


def sauvegarder_image_unicode(chemin_image, image):
    """Sauvegarde une image même si le chemin contient des accents."""

    extension = chemin_image.suffix.lower()

    if extension not in EXTENSIONS_IMAGES:
        extension = ".png"

    succes, donnees = cv2.imencode(extension, image)

    if not succes:
        raise IOError(
            f"Impossible d'encoder l'image : {chemin_image}"
        )

    donnees.tofile(str(chemin_image))


# 5. CHARGEMENT DES FICHIERS DE CALIBRATION

def charger_json(chemin):
    """Charge un fichier JSON et vérifie son existence."""

    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier obligatoire introuvable : {chemin}"
        )

    with open(chemin, "r", encoding="utf-8") as fichier:
        return json.load(fichier)


def charger_configuration_calibration():
    """Charge les ROI et les références obtenues avec le Programme 2."""

    donnees_roi = charger_json(FICHIER_ROI)
    donnees_references = charger_json(FICHIER_REFERENCES)

    return donnees_roi, donnees_references

# 6. OUVERTURE ET CAPTURE DES CAMÉRAS

def ouvrir_camera(index_camera, nom_camera):
    """Ouvre une caméra et demande la résolution définie plus haut."""

    print(
        f"Ouverture caméra {nom_camera}, index {index_camera}..."
    )

    # Backend DirectShow sous Windows.
    camera = cv2.VideoCapture(
        index_camera,
        cv2.CAP_DSHOW
    )

    # Deuxième tentative sans backend imposé.
    if not camera.isOpened():
        camera = cv2.VideoCapture(index_camera)

    if not camera.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la caméra {nom_camera}."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        LARGEUR_CAMERA
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HAUTEUR_CAMERA
    )

    # Désactivation de l'autofocus lorsque le pilote le permet.
    camera.set(
        cv2.CAP_PROP_AUTOFOCUS,
        0
    )

    time.sleep(0.5)

    largeur_reelle = int(
        camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    hauteur_reelle = int(
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(
        f"{nom_camera} ouverte : "
        f"{largeur_reelle} x {hauteur_reelle} px"
    )

    return camera


def capturer_image(camera, nom_camera):
    """Capture une image après avoir supprimé quelques trames."""

    for _ in range(NOMBRE_IMAGES_A_JETER):
        camera.read()

    succes, image = camera.read()

    if not succes or image is None:
        raise RuntimeError(
            f"Erreur de capture avec la caméra {nom_camera}."
        )

    return image


def capturer_deux_faces(camera_haut, camera_bas):
    """Capture la face avant puis la face arrière."""

    image_avant = capturer_image(
        camera_haut,
        "haute / face avant"
    )

    image_arriere = capturer_image(
        camera_bas,
        "basse / face arrière"
    )

    return image_avant, image_arriere


# 7. DÉTECTION AUTOMATIQUE DU TAMBOUR

def detecter_tambour_hough(image):
    """
    Détecte le contour circulaire principal du tambour.

    Cette fonction reprend la logique utilisée dans le Programme 1.
    """

    hauteur_originale, largeur_originale = image.shape[:2]
    dimension_max = max(
        hauteur_originale,
        largeur_originale
    )

    facteur = min(
        1.0,
        1200.0 / float(dimension_max)
    )

    largeur_travail = int(
        round(largeur_originale * facteur)
    )

    hauteur_travail = int(
        round(hauteur_originale * facteur)
    )

    image_travail = cv2.resize(
        image,
        (largeur_travail, hauteur_travail),
        interpolation=cv2.INTER_AREA
    )

    gris = cv2.cvtColor(
        image_travail,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gris = clahe.apply(gris)
    gris = cv2.medianBlur(gris, 9)

    dimension_min = min(
        hauteur_travail,
        largeur_travail
    )

    rayon_min = int(
        round(0.25 * dimension_min)
    )

    rayon_max = int(
        round(0.49 * dimension_min)
    )

    centre_image_x = largeur_travail / 2.0
    centre_image_y = hauteur_travail / 2.0

    candidats = []

    for seuil_hough in (65, 58, 52, 46, 40, 34):

        cercles = cv2.HoughCircles(
            gris,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(round(0.35 * dimension_min)),
            param1=130,
            param2=seuil_hough,
            minRadius=rayon_min,
            maxRadius=rayon_max
        )

        if cercles is None:
            continue

        for x, y, rayon in cercles[0]:

            marge_bord = 0.90 * rayon

            if (
                x < marge_bord
                or y < marge_bord
                or x > largeur_travail - marge_bord
                or y > hauteur_travail - marge_bord
            ):
                continue

            distance = np.hypot(
                x - centre_image_x,
                y - centre_image_y
            )

            score = (
                3.0 * rayon / dimension_min
                - 1.15 * distance / dimension_min
                + 0.003 * seuil_hough
            )

            candidats.append(
                (score, x, y, rayon, seuil_hough)
            )

    if not candidats:
        return None

    candidats.sort(
        key=lambda element: element[0],
        reverse=True
    )

    score, x, y, rayon, seuil = candidats[0]

    return {
        "centre_x": int(round(x / facteur)),
        "centre_y": int(round(y / facteur)),
        "rayon": int(round(rayon / facteur)),
        "score": float(score),
        "methode": f"Hough param2={seuil}"
    }


# 8. CENTRAGE ET FOND NOIR


def extraire_et_centrer_tambour(image, detection):
    """
    Applique exactement la normalisation géométrique du Programme 1.
    """

    centre_x = int(detection["centre_x"])
    centre_y = int(detection["centre_y"])
    rayon = max(10, int(detection["rayon"]))

    hauteur_image, largeur_image = image.shape[:2]

    rayon_recadrage = int(
        round(
            rayon * (1.0 + MARGE_AUTOUR_PIECE)
        )
    )

    x1 = centre_x - rayon_recadrage
    y1 = centre_y - rayon_recadrage
    x2 = centre_x + rayon_recadrage
    y2 = centre_y + rayon_recadrage

    taille_carre = 2 * rayon_recadrage

    carre_noir = np.zeros(
        (taille_carre, taille_carre, 3),
        dtype=np.uint8
    )

    source_x1 = max(0, x1)
    source_y1 = max(0, y1)
    source_x2 = min(largeur_image, x2)
    source_y2 = min(hauteur_image, y2)

    destination_x1 = source_x1 - x1
    destination_y1 = source_y1 - y1
    destination_x2 = (
        destination_x1
        + source_x2
        - source_x1
    )
    destination_y2 = (
        destination_y1
        + source_y2
        - source_y1
    )

    carre_noir[
        destination_y1:destination_y2,
        destination_x1:destination_x2
    ] = image[
        source_y1:source_y2,
        source_x1:source_x2
    ]

    masque = np.zeros(
        (taille_carre, taille_carre),
        dtype=np.uint8
    )

    cv2.circle(
        masque,
        (rayon_recadrage, rayon_recadrage),
        rayon,
        255,
        -1,
        lineType=cv2.LINE_AA
    )

    piece_fond_noir = cv2.bitwise_and(
        carre_noir,
        carre_noir,
        mask=masque
    )

    image_finale = cv2.resize(
        piece_fond_noir,
        (
            TAILLE_IMAGE_NORMALISEE,
            TAILLE_IMAGE_NORMALISEE
        ),
        interpolation=cv2.INTER_AREA
    )

    return image_finale


def normaliser_image_si_necessaire(image):
    """
    Centre l'image brute.
    Si l'image fait déjà 3000 x 3000 px et possède un fond noir,
    elle peut être utilisée directement en mode test.
    """

    hauteur, largeur = image.shape[:2]

    # Les images du Programme 1 sont déjà normalisées.
    if (
        hauteur == TAILLE_IMAGE_NORMALISEE
        and largeur == TAILLE_IMAGE_NORMALISEE
    ):
        return image.copy(), {
            "methode": "Image déjà normalisée",
            "centre_x": largeur // 2,
            "centre_y": hauteur // 2,
            "rayon": int(
                0.94
                * min(largeur, hauteur)
                / 2
            )
        }

    detection = detecter_tambour_hough(image)

    if detection is None:
        raise RuntimeError(
            "Tambour principal non détecté."
        )

    image_centree = extraire_et_centrer_tambour(
        image,
        detection
    )

    return image_centree, detection


# 9. ROI NORMALISÉES

def denormaliser_roi(
    roi_normalisee,
    largeur_image,
    hauteur_image
):
    """Convertit une ROI normalisée vers les pixels."""

    x = int(
        round(
            roi_normalisee["x_norm"]
            * largeur_image
        )
    )

    y = int(
        round(
            roi_normalisee["y_norm"]
            * hauteur_image
        )
    )

    largeur = int(
        round(
            roi_normalisee["largeur_norm"]
            * largeur_image
        )
    )

    hauteur = int(
        round(
            roi_normalisee["hauteur_norm"]
            * hauteur_image
        )
    )

    return x, y, largeur, hauteur


def extraire_roi(image, roi):
    """Extrait une ROI en sécurisant ses limites."""

    x, y, largeur, hauteur = roi
    hauteur_image, largeur_image = image.shape[:2]

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(
        largeur_image,
        int(x + largeur)
    )
    y2 = min(
        hauteur_image,
        int(y + hauteur)
    )

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"ROI invalide : {roi}"
        )

    return image[y1:y2, x1:x2]


# 10. CALCUL DES CARACTÉRISTIQUES

def calculer_caracteristiques_roi(roi_bgr):
    """
    Calcule les mêmes caractéristiques que le Programme 2.
    """

    gris = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2HSV
    )

    _, saturation, luminosite = cv2.split(hsv)

    gris_moyenne = float(np.mean(gris))
    gris_ecart_type = float(np.std(gris))
    gris_mediane = float(np.median(gris))
    gris_p05 = float(np.percentile(gris, 5))
    gris_p95 = float(np.percentile(gris, 95))
    contraste_p90 = float(gris_p95 - gris_p05)

    saturation_moyenne = float(
        np.mean(saturation)
    )
    saturation_mediane = float(
        np.median(saturation)
    )
    luminosite_moyenne = float(
        np.mean(luminosite)
    )

    masque_peinture = np.zeros_like(
        saturation,
        dtype=np.uint8
    )

    masque_peinture[
        (saturation > 45)
        & (luminosite > 35)
    ] = 255

    ratio_pixels_colores = float(
        np.mean(masque_peinture > 0)
    )

    gris_lisse = cv2.GaussianBlur(
        gris,
        (5, 5),
        0
    )

    contours = cv2.Canny(
        gris_lisse,
        50,
        150
    )

    densite_contours = float(
        np.mean(contours > 0)
    )

    nombre_pixels_contours = int(
        np.sum(contours > 0)
    )

    laplacien = cv2.Laplacian(
        gris,
        cv2.CV_64F
    )

    variance_laplacien = float(
        np.var(laplacien)
    )

    laplacien_absolu = np.uint8(
        np.clip(
            np.abs(laplacien),
            0,
            255
        )
    )

    _, masque_texture = cv2.threshold(
        laplacien_absolu,
        45,
        255,
        cv2.THRESH_BINARY
    )

    ratio_texture = float(
        np.mean(masque_texture > 0)
    )

    ratio_pixels_sombres = float(
        np.mean(gris < 40)
    )

    ratio_pixels_clairs = float(
        np.mean(gris > 215)
    )

    return {
        "gris_moyenne": gris_moyenne,
        "gris_ecart_type": gris_ecart_type,
        "gris_mediane": gris_mediane,
        "gris_p05": gris_p05,
        "gris_p95": gris_p95,
        "contraste_p90": contraste_p90,
        "saturation_moyenne": saturation_moyenne,
        "saturation_mediane": saturation_mediane,
        "luminosite_moyenne": luminosite_moyenne,
        "ratio_pixels_colores": ratio_pixels_colores,
        "densite_contours": densite_contours,
        "nombre_pixels_contours": nombre_pixels_contours,
        "variance_laplacien": variance_laplacien,
        "ratio_texture": ratio_texture,
        "ratio_pixels_sombres": ratio_pixels_sombres,
        "ratio_pixels_clairs": ratio_pixels_clairs
    }


# 11. COMPARAISON AUX RÉFÉRENCES

def obtenir_limites_reference(
    donnees_references,
    face,
    nom_roi,
    caracteristique
):
    """
    Cherche la plage de référence d'une caractéristique.
    """

    cle_roi = f"{face}/{nom_roi}"

    references = donnees_references.get(
        "references",
        {}
    )

    donnees_roi = references.get(cle_roi)

    if donnees_roi is None:
        return None

    donnees_caracteristique = donnees_roi.get(
        caracteristique
    )

    if donnees_caracteristique is None:
        return None

    # Priorité à un vrai seuil OK/NOK s'il existe.
    seuil_supervise = (
        donnees_caracteristique
        .get("seuil_OK_NOK_suggere")
    )

    if seuil_supervise is not None:
        return {
            "type": "seuil_supervise",
            "seuil": float(
                seuil_supervise["seuil_suggere"]
            ),
            "regle": seuil_supervise["regle"],
            "groupes_clairement_separes": bool(
                seuil_supervise[
                    "groupes_clairement_separes"
                ]
            )
        }

    # Sinon, utilisation de la plage provisoire des pièces OK.
    statistiques_ok = (
        donnees_caracteristique
        .get("statistiques_OK")
    )

    if statistiques_ok is None:
        return None

    plage = statistiques_ok.get(
        "plage_reference_OK_provisoire"
    )

    if plage is None:
        return None

    borne_basse = float(
        plage["borne_basse"]
    )
    borne_haute = float(
        plage["borne_haute"]
    )

    largeur_plage = max(
        borne_haute - borne_basse,
        abs(borne_haute) * 0.02,
        1e-6
    )

    # Élargissement de la plage
    borne_basse -= (
        MARGE_SECURITE_REFERENCES
        * largeur_plage
    )

    borne_haute += (
        MARGE_SECURITE_REFERENCES
        * largeur_plage
    )

    return {
        "type": "plage_OK_provisoire",
        "borne_basse": borne_basse,
        "borne_haute": borne_haute
    }


def evaluer_valeur(
    valeur,
    limites
):
    """Évalue une valeur par rapport à une plage ou à un seuil."""

    if limites is None:
        return {
            "conforme": None,
            "message": "Référence absente"
        }

    if limites["type"] == "plage_OK_provisoire":

        conforme = (
            limites["borne_basse"]
            <= valeur
            <= limites["borne_haute"]
        )

        return {
            "conforme": bool(conforme),
            "message": (
                f"{limites['borne_basse']:.6g} "
                f"<= {valeur:.6g} <= "
                f"{limites['borne_haute']:.6g}"
            )
        }

    seuil = limites["seuil"]
    regle = limites["regle"]

    if regle == "OK_si_valeur_superieure_ou_egale":
        conforme = valeur >= seuil
        symbole = ">="

    else:
        conforme = valeur <= seuil
        symbole = "<="

    return {
        "conforme": bool(conforme),
        "message": (
            f"{valeur:.6g} {symbole} "
            f"{seuil:.6g}"
        )
    }


# 12. ANALYSE D'UNE FACE

def analyser_face(
    image_centree,
    face,
    rois_normalisees,
    donnees_references
):
    """
    Analyse toutes les ROI utiles d'une face.
    """

    hauteur, largeur = image_centree.shape[:2]

    resultats_roi = {}
    rois_pixels = {}
    anomalies = []

    for nom_roi, roi_normalisee in rois_normalisees.items():

        # La ROI piece_complete sert surtout de repère visuel.
        if nom_roi == "piece_complete":
            continue

        roi_pixels = denormaliser_roi(
            roi_normalisee,
            largeur,
            hauteur
        )

        rois_pixels[nom_roi] = roi_pixels

        roi_bgr = extraire_roi(
            image_centree,
            roi_pixels
        )

        caracteristiques = (
            calculer_caracteristiques_roi(
                roi_bgr
            )
        )

        controles_roi = {}

        noms_caracteristiques = (
            CARACTERISTIQUES_DECISION.get(
                nom_roi,
                []
            )
        )

        for nom_caracteristique in noms_caracteristiques:

            valeur = caracteristiques[
                nom_caracteristique
            ]

            limites = obtenir_limites_reference(
                donnees_references,
                face,
                nom_roi,
                nom_caracteristique
            )

            evaluation = evaluer_valeur(
                valeur,
                limites
            )

            controles_roi[nom_caracteristique] = {
                "valeur": valeur,
                "limites": limites,
                "evaluation": evaluation
            }

            if evaluation["conforme"] is False:
                anomalies.append(
                    {
                        "face": face,
                        "roi": nom_roi,
                        "caracteristique": nom_caracteristique,
                        "valeur": valeur,
                        "message": evaluation["message"]
                    }
                )

        resultats_roi[nom_roi] = {
            "roi_pixels": list(roi_pixels),
            "caracteristiques": caracteristiques,
            "controles_decision": controles_roi
        }

    return resultats_roi, anomalies, rois_pixels


# 13. DÉCISION FINALE

def prendre_decision_finale(
    anomalies,
    donnees_references
):
    """
    Produit la décision globale sans commande de tampon.
    """

    nombre_anomalies = len(anomalies)

    contient_nok = bool(
        donnees_references
        .get("base_actuelle", {})
        .get("contient_images_NOK", False)
    )

    if nombre_anomalies <= NOMBRE_ANOMALIES_TOLEREES:
        return {
            "decision": "OK",
            "cause": (
                "Toutes les caractéristiques contrôlées "
                "sont dans les limites de référence."
            ),
            "nombre_anomalies": nombre_anomalies,
            "niveau_confiance": (
                "provisoire"
                if not contient_nok
                else "calibré_OK_NOK"
            )
        }

    if not contient_nok:

        decision = (
            "A_VERIFIER"
            if UTILISER_STATUT_A_VERIFIER_SANS_NOK
            else "NOK_PROVISOIRE"
        )

        cause = (
            f"{nombre_anomalies} caractéristique(s) "
            "hors des plages des pièces conformes. "
            "La base ne contient pas encore d'images NOK."
        )

        niveau = "provisoire_sans_images_NOK"

    else:
        decision = "NOK"
        cause = (
            f"{nombre_anomalies} contrôle(s) "
            "non conforme(s)."
        )
        niveau = "calibré_OK_NOK"

    return {
        "decision": decision,
        "cause": cause,
        "nombre_anomalies": nombre_anomalies,
        "niveau_confiance": niveau
    }


# 14. ANNOTATION DES IMAGES

def couleur_decision(decision):
    """Retourne une couleur OpenCV selon la décision."""

    if decision == "OK":
        return (0, 255, 0)

    if decision in ("A_VERIFIER", "NOK_PROVISOIRE"):
        return (0, 165, 255)

    return (0, 0, 255)


def annoter_image(
    image,
    rois_pixels,
    resultats_roi,
    decision,
    face
):
    """
    Dessine les ROI et la décision sur l'image.
    """

    sortie = image.copy()
    couleur_globale = couleur_decision(
        decision
    )

    for nom_roi, roi in rois_pixels.items():

        x, y, largeur, hauteur = roi

        controles = (
            resultats_roi
            .get(nom_roi, {})
            .get("controles_decision", {})
        )

        anomalie_roi = any(
            controle["evaluation"]["conforme"] is False
            for controle in controles.values()
        )

        couleur_roi = (
            (0, 0, 255)
            if anomalie_roi
            else (0, 255, 0)
        )

        cv2.rectangle(
            sortie,
            (x, y),
            (x + largeur, y + hauteur),
            couleur_roi,
            4
        )

        cv2.putText(
            sortie,
            nom_roi,
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            couleur_roi,
            2
        )

    cv2.rectangle(
        sortie,
        (0, 0),
        (TAILLE_IMAGE_NORMALISEE, 115),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        sortie,
        f"{face} - DECISION : {decision}",
        (40, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        couleur_globale,
        4
    )

    return sortie


# 15. ENREGISTREMENT DES RÉSULTATS


def sauvegarder_resultat_csv(
    id_piece,
    decision,
    cause,
    nombre_anomalies,
    niveau_confiance,
    chemin_avant,
    chemin_arriere
):
    """Ajoute une ligne au fichier CSV global."""

    existe = FICHIER_RESULTATS_CSV.exists()

    with open(
        FICHIER_RESULTATS_CSV,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as fichier:

        writer = csv.writer(
            fichier,
            delimiter=";"
        )

        if not existe:
            writer.writerow(
                [
                    "id_piece",
                    "date_heure",
                    "decision",
                    "cause",
                    "nombre_anomalies",
                    "niveau_confiance",
                    "image_face_avant",
                    "image_face_arriere"
                ]
            )

        writer.writerow(
            [
                id_piece,
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                decision,
                cause,
                nombre_anomalies,
                niveau_confiance,
                str(chemin_avant),
                str(chemin_arriere)
            ]
        )


def sauvegarder_details_json(
    id_piece,
    details
):
    """Sauvegarde tous les détails techniques d'une inspection."""

    chemin = (
        DOSSIER_DETAILS_JSON
        / f"{id_piece}_details.json"
    )

    with open(
        chemin,
        "w",
        encoding="utf-8"
    ) as fichier:

        json.dump(
            details,
            fichier,
            indent=4,
            ensure_ascii=False
        )

    return chemin


# 16. AFFICHAGE

def afficher_image(titre, image):
    """Affiche une image OpenCV redimensionnée pour l'écran."""

    if not AFFICHER_IMAGES:
        return

    hauteur, largeur = image.shape[:2]
    facteur = min(
        1.0,
        1000.0 / max(hauteur, largeur)
    )

    apercu = cv2.resize(
        image,
        (
            int(round(largeur * facteur)),
            int(round(hauteur * facteur))
        ),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow(titre, apercu)


# 17. PIPELINE COMPLET D'UNE PIÈCE

def analyser_piece(
    image_avant_brute,
    image_arriere_brute,
    id_piece,
    donnees_roi,
    donnees_references
):
    """
    Exécute toutes les opérations sur une pièce.
    """

    print("\n============================================================")
    print(f"ANALYSE DE LA PIÈCE : {id_piece}")
    print("============================================================")

    horodatage = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    chemin_brut_avant = (
        DOSSIER_IMAGES_BRUTES
        / f"{id_piece}_face_avant_{horodatage}.png"
    )

    chemin_brut_arriere = (
        DOSSIER_IMAGES_BRUTES
        / f"{id_piece}_face_arriere_{horodatage}.png"
    )

    if SAUVEGARDER_IMAGES:
        sauvegarder_image_unicode(
            chemin_brut_avant,
            image_avant_brute
        )
        sauvegarder_image_unicode(
            chemin_brut_arriere,
            image_arriere_brute
        )

    image_avant_centree, detection_avant = (
        normaliser_image_si_necessaire(
            image_avant_brute
        )
    )

    image_arriere_centree, detection_arriere = (
        normaliser_image_si_necessaire(
            image_arriere_brute
        )
    )

    chemin_centre_avant = (
        DOSSIER_IMAGES_CENTREES
        / f"{id_piece}_face_avant_centree.png"
    )

    chemin_centre_arriere = (
        DOSSIER_IMAGES_CENTREES
        / f"{id_piece}_face_arriere_centree.png"
    )

    if SAUVEGARDER_IMAGES:
        sauvegarder_image_unicode(
            chemin_centre_avant,
            image_avant_centree
        )
        sauvegarder_image_unicode(
            chemin_centre_arriere,
            image_arriere_centree
        )

    rois_avant = (
        donnees_roi["face_avant"]["normalisees"]
    )

    rois_arriere = (
        donnees_roi["face_arriere"]["normalisees"]
    )

    resultats_avant, anomalies_avant, rois_pixels_avant = (
        analyser_face(
            image_avant_centree,
            "face_avant",
            rois_avant,
            donnees_references
        )
    )

    resultats_arriere, anomalies_arriere, rois_pixels_arriere = (
        analyser_face(
            image_arriere_centree,
            "face_arriere",
            rois_arriere,
            donnees_references
        )
    )

    anomalies = (
        anomalies_avant
        + anomalies_arriere
    )

    resultat_decision = prendre_decision_finale(
        anomalies,
        donnees_references
    )

    decision = resultat_decision["decision"]
    cause = resultat_decision["cause"]

    image_avant_annotee = annoter_image(
        image_avant_centree,
        rois_pixels_avant,
        resultats_avant,
        decision,
        "FACE AVANT"
    )

    image_arriere_annotee = annoter_image(
        image_arriere_centree,
        rois_pixels_arriere,
        resultats_arriere,
        decision,
        "FACE ARRIERE"
    )

    chemin_annote_avant = (
        DOSSIER_IMAGES_ANNOTEES
        / f"{id_piece}_face_avant_resultat.png"
    )

    chemin_annote_arriere = (
        DOSSIER_IMAGES_ANNOTEES
        / f"{id_piece}_face_arriere_resultat.png"
    )

    if SAUVEGARDER_IMAGES:
        sauvegarder_image_unicode(
            chemin_annote_avant,
            image_avant_annotee
        )

        sauvegarder_image_unicode(
            chemin_annote_arriere,
            image_arriere_annotee
        )

    details = {
        "id_piece": id_piece,
        "date_heure": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "decision": resultat_decision,
        "detection_face_avant": detection_avant,
        "detection_face_arriere": detection_arriere,
        "resultats_face_avant": resultats_avant,
        "resultats_face_arriere": resultats_arriere,
        "anomalies": anomalies,
        "avertissement": (
            "Décision provisoire tant que la calibration "
            "ne contient pas d'images NOK réelles."
        )
    }

    chemin_json = sauvegarder_details_json(
        id_piece,
        details
    )

    sauvegarder_resultat_csv(
        id_piece,
        decision,
        cause,
        resultat_decision["nombre_anomalies"],
        resultat_decision["niveau_confiance"],
        chemin_annote_avant,
        chemin_annote_arriere
    )

    print(f"Décision           : {decision}")
    print(f"Cause              : {cause}")
    print(
        f"Nombre d'anomalies : "
        f"{resultat_decision['nombre_anomalies']}"
    )

    if anomalies:
        print("\nContrôles hors référence :")

        for anomalie in anomalies:
            print(
                f"- {anomalie['face']} / "
                f"{anomalie['roi']} / "
                f"{anomalie['caracteristique']} : "
                f"{anomalie['message']}"
            )

    print(f"Détails JSON       : {chemin_json}")

    afficher_image(
        "Résultat face avant",
        image_avant_annotee
    )

    afficher_image(
        "Résultat face arrière",
        image_arriere_annotee
    )

    if AFFICHER_IMAGES:
        print(
            "Appuyer sur une touche dans une fenêtre "
            "pour continuer."
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return details


# 18. MODE TEST

def mode_test_images(
    donnees_roi,
    donnees_references
):
    """Teste le pipeline sur deux images enregistrées."""

    image_avant = lire_image_unicode(
        PATH_TEST_FACE_AVANT
    )

    image_arriere = lire_image_unicode(
        PATH_TEST_FACE_ARRIERE
    )

    analyser_piece(
        image_avant,
        image_arriere,
        "piece_test_0001",
        donnees_roi,
        donnees_references
    )


# 19. MODE DEUX CAMÉRAS

def attendre_piece_simulee(numero_piece):
    """
    Simulation du déclenchement d'une nouvelle pièce.

    Plus tard, cette fonction pourra être remplacée par :
        - une entrée automate ;
        - un capteur de présence ;
        - un bouton opérateur ;
        - une commande réseau.
    """

    reponse = input(
        f"\nEntrée = analyser la pièce {numero_piece}, "
        "q = quitter : "
    )

    return reponse.strip().lower() != "q"


def mode_deux_cameras(
    donnees_roi,
    donnees_references
):
    """Mode d'acquisition réelle avec deux caméras."""

    camera_haut = None
    camera_bas = None

    try:
        camera_haut = ouvrir_camera(
            CAMERA_HAUT_INDEX,
            "haute"
        )

        camera_bas = ouvrir_camera(
            CAMERA_BAS_INDEX,
            "basse"
        )

        numero_piece = 1

        while True:

            if not attendre_piece_simulee(
                numero_piece
            ):
                break

            id_piece = (
                f"piece_{numero_piece:04d}"
            )

            print(
                "Convoyeur supposé arrêté ou stabilisé."
            )

            image_avant, image_arriere = (
                capturer_deux_faces(
                    camera_haut,
                    camera_bas
                )
            )

            analyser_piece(
                image_avant,
                image_arriere,
                id_piece,
                donnees_roi,
                donnees_references
            )

            numero_piece += 1

    finally:
        if camera_haut is not None:
            camera_haut.release()

        if camera_bas is not None:
            camera_bas.release()

        cv2.destroyAllWindows()


# 20. PROGRAMME PRINCIPAL


def main():
    """Charge la calibration puis lance le mode choisi."""

    print("============================================================")
    print(" PROGRAMME 3 — INSPECTION FINALE À DEUX CAMÉRAS")
    print("============================================================")
    print("Aucun signal de tampon n'est utilisé.")
    print("La sortie est une décision logicielle et un rapport.")

    donnees_roi, donnees_references = (
        charger_configuration_calibration()
    )

    contient_nok = bool(
        donnees_references
        .get("base_actuelle", {})
        .get("contient_images_NOK", False)
    )

    print(f"Fichier ROI         : {FICHIER_ROI}")
    print(f"Fichier références  : {FICHIER_REFERENCES}")
    print(f"Calibration avec NOK: {contient_nok}")

    if not contient_nok:
        print(
            "\nATTENTION : la calibration actuelle "
            "ne contient que des pièces conformes."
        )
        print(
            "Les décisions hors plage seront donc "
            "classées A_VERIFIER."
        )

    if MODE_TEST_IMAGES:
        print("\nMode choisi : TEST AVEC IMAGES")
        mode_test_images(
            donnees_roi,
            donnees_references
        )

    else:
        print("\nMode choisi : DEUX CAMÉRAS")
        mode_deux_cameras(
            donnees_roi,
            donnees_references
        )


# 21. EXÉCUTION

if __name__ == "__main__":
    main()

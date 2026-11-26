PROGRAMME 1 — PRÉPARATION DE LA BASE D'IMAGES DES TAMBOURS MTV9

Ce programme prépare les photos prises avec mon téléphone avant la calibration des ROI.

Pour chaque photo, il :
    1. cherche automatiquement le tambour principal ;
    2. permet de vérifier ou de corriger la détection ;
    3. extrait uniquement le tambour ;
    4. remplace tout ce qui se trouve autour par un fond noir ;
    5. centre la pièce dans une image carrée ;
    6. redimensionne toutes les images au même format ;
    7. sauvegarde les images nettoyées et un rapport CSV.
"""

# IMPORTATION DES BIBLIOTHÈQUES

import cv2

import numpy as np

import csv

from pathlib import Path

from datetime import datetime


# 1. CHEMINS DES DOSSIERS

DOSSIER_BASE = Path(__file__).resolve().parents[1] / "data"

DOSSIER_ENTREE_AVANT = DOSSIER_BASE / "Face_Avant"

DOSSIER_ENTREE_ARRIERE = DOSSIER_BASE / "Face_Arrière"

DOSSIER_SORTIE_AVANT = DOSSIER_BASE / "Face_Avant_Centree"

DOSSIER_SORTIE_ARRIERE = DOSSIER_BASE / "Face_Arriere_Centree"

DOSSIER_DIAGNOSTIC = DOSSIER_BASE / "Diagnostic_Detection"

FICHIER_RAPPORT = DOSSIER_BASE / "rapport_pretraitement.csv"

DOSSIER_SORTIE_AVANT.mkdir(parents=True, exist_ok=True)
DOSSIER_SORTIE_ARRIERE.mkdir(parents=True, exist_ok=True)
DOSSIER_DIAGNOSTIC.mkdir(parents=True, exist_ok=True)

# 2. PARAMÈTRES DU PRÉTRAITEMENT

TAILLE_SORTIE = 3000

MARGE_AUTOUR_PIECE = 0.06

VALIDATION_INTERACTIVE = True

LARGEUR_APERCU = 1100

EXTENSIONS_IMAGES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# 3. LECTURE DES IMAGES, Y COMPRIS LES CHEMINS AVEC ACCENTS


def lire_image_unicode(chemin_image):
    """
    Lit une image même lorsque le chemin Windows contient des accents.

    cv2.imread peut parfois échouer avec certains caractères Unicode sous Windows.
    La combinaison np.fromfile + cv2.imdecode est plus robuste.
    """

    donnees = np.fromfile(str(chemin_image), dtype=np.uint8)

    image = cv2.imdecode(donnees, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Impossible de lire l'image : {chemin_image}"
        )

    return image


def sauvegarder_image_unicode(chemin_image, image):
    """
    Sauvegarde une image même lorsque le chemin contient des accents.
    """

    extension = chemin_image.suffix.lower()

    if extension not in EXTENSIONS_IMAGES:
        extension = ".png"

    succes, donnees_encodees = cv2.imencode(extension, image)

    if not succes:
        raise IOError(
            f"Impossible d'encoder l'image : {chemin_image}"
        )

    donnees_encodees.tofile(str(chemin_image))


# 4. LISTE DES IMAGES D'UN DOSSIER

def lister_images(dossier):
    """
    Retourne la liste triée de toutes les images présentes dans un dossier.
    """

    if not dossier.exists():
        raise FileNotFoundError(
            f"Le dossier n'existe pas : {dossier}"
        )

    images = []

    for chemin in dossier.iterdir():

        if chemin.is_file() and chemin.suffix.lower() in EXTENSIONS_IMAGES:
            images.append(chemin)

    return sorted(images)

# 5. CRÉATION D'UN APERÇU ADAPTÉ À L'ÉCRAN

def creer_apercu(image, largeur_max=LARGEUR_APERCU):
    """
    Réduit uniquement l'image affichée à l'écran.
    L'image originale n'est pas modifiée.
    """

    hauteur, largeur = image.shape[:2]

    if largeur <= largeur_max:
        return image.copy(), 1.0

    facteur = largeur_max / float(largeur)

    nouvelle_hauteur = int(round(hauteur * facteur))

    apercu = cv2.resize(
        image,
        (largeur_max, nouvelle_hauteur),
        interpolation=cv2.INTER_AREA
    )

    return apercu, facteur


# 6. DÉTECTION AUTOMATIQUE DU TAMBOUR

def detecter_tambour_hough(image):
    """
    Détecte le grand cercle correspondant au tambour principal.

    La méthode est adaptée aux photos transmises :
        - tambour métallique circulaire ;
        - fond plus sombre ;
        - plusieurs pièces parfois visibles autour ;
        - tambour principal généralement proche de la zone centrale.

    Retour :
        centre_x, centre_y, rayon, score, methode
    ou None si aucune détection fiable n'est trouvée.
    """

    hauteur_originale, largeur_originale = image.shape[:2]

    # Réduction temporaire de l'image pour accélérer la détection.
    dimension_max = max(hauteur_originale, largeur_originale)

    facteur = min(1.0, 1200.0 / float(dimension_max))

    largeur_travail = int(round(largeur_originale * facteur))
    hauteur_travail = int(round(hauteur_originale * facteur))

    image_travail = cv2.resize(
        image,
        (largeur_travail, hauteur_travail),
        interpolation=cv2.INTER_AREA
    )

    gris = cv2.cvtColor(image_travail, cv2.COLOR_BGR2GRAY)

    # Amélioration du contraste avant la détection des cercles.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gris = clahe.apply(gris)

    gris = cv2.medianBlur(gris, 9)

    dimension_min = min(hauteur_travail, largeur_travail)

    rayon_min = int(round(0.25 * dimension_min))
    rayon_max = int(round(0.49 * dimension_min))

    centre_image_x = largeur_travail / 2.0
    centre_image_y = hauteur_travail / 2.0

    candidats = []

    # Plusieurs seuils sont testés pour obtenir une détection plus robuste.
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

        cercles = cercles[0]

        for x, y, rayon in cercles:

            marge_bord = 0.90 * rayon

            if (
                x < marge_bord
                or y < marge_bord
                or x > largeur_travail - marge_bord
                or y > hauteur_travail - marge_bord
            ):
                continue

            distance_centre = np.hypot(
                x - centre_image_x,
                y - centre_image_y
            )

            distance_normalisee = distance_centre / float(dimension_min)

            rayon_normalise = rayon / float(dimension_min)

            score = (
                3.0 * rayon_normalise
                - 1.15 * distance_normalisee
                + 0.003 * seuil_hough
            )

            candidats.append(
                (score, float(x), float(y), float(rayon), seuil_hough)
            )


    if not candidats:
        return None

    candidats.sort(key=lambda element: element[0], reverse=True)

    meilleur_score, x, y, rayon, seuil_utilise = candidats[0]

    x_original = x / facteur
    y_original = y / facteur
    rayon_original = rayon / facteur

    return {
        "centre_x": int(round(x_original)),
        "centre_y": int(round(y_original)),
        "rayon": int(round(rayon_original)),
        "score": float(meilleur_score),
        "methode": f"Hough automatique, param2={seuil_utilise}"
    }


# 7. SÉLECTION MANUELLE EN CAS D'ÉCHEC

def selection_manuelle_tambour(image, nom_fenetre):
    """
    Permet de sélectionner manuellement un carré autour du tambour.

    Il faut :
        1. tracer un rectangle qui entoure précisément le cercle extérieur ;
        2. appuyer sur Entrée ou Espace pour valider ;
        3. appuyer sur C ou Échap pour annuler.

    Le centre et le rayon sont ensuite calculés à partir du rectangle.
    """

    apercu, facteur = creer_apercu(image)

    cv2.putText(
        apercu,
        "Encadrer le tambour principal puis valider avec ENTREE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    x, y, largeur, hauteur = cv2.selectROI(
        nom_fenetre,
        apercu,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow(nom_fenetre)

    if largeur == 0 or hauteur == 0:
        return None

    x_original = x / facteur
    y_original = y / facteur
    largeur_originale = largeur / facteur
    hauteur_originale = hauteur / facteur

    diametre = min(largeur_originale, hauteur_originale)

    rayon = diametre / 2.0

    centre_x = x_original + largeur_originale / 2.0
    centre_y = y_original + hauteur_originale / 2.0

    return {
        "centre_x": int(round(centre_x)),
        "centre_y": int(round(centre_y)),
        "rayon": int(round(rayon)),
        "score": None,
        "methode": "Sélection manuelle"
    }


# 8. AFFICHAGE ET VALIDATION DE LA DÉTECTION

def valider_detection(image, detection, nom_image):
    """
    Affiche le cercle détecté et demande une validation.

    Touches :
        A ou Entrée : accepter la détection ;
        M            : corriger manuellement ;
        S            : ignorer cette image ;
        Q            : arrêter le programme.
    """

    # En cas d'échec, passage à la sélection manuelle.
    if detection is None:
        print(
            f"[ATTENTION] Détection automatique impossible pour {nom_image}."
        )

        detection_manuelle = selection_manuelle_tambour(
            image,
            f"Selection manuelle - {nom_image}"
        )

        return detection_manuelle, "continuer"

    if not VALIDATION_INTERACTIVE:
        return detection, "continuer"

    diagnostic = image.copy()

    centre_x = detection["centre_x"]
    centre_y = detection["centre_y"]
    rayon = detection["rayon"]

    cv2.circle(
        diagnostic,
        (centre_x, centre_y),
        rayon,
        (0, 255, 0),
        8
    )

    cv2.circle(
        diagnostic,
        (centre_x, centre_y),
        15,
        (0, 0, 255),
        -1
    )

    apercu, _ = creer_apercu(diagnostic)

    cv2.putText(
        apercu,
        "A/ENTREE = accepter | M = manuel | S = ignorer | Q = quitter",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        apercu,
        detection["methode"],
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 0),
        2
    )

    nom_fenetre = f"Validation - {nom_image}"

    cv2.imshow(nom_fenetre, apercu)

    touche = cv2.waitKey(0) & 0xFF

    cv2.destroyWindow(nom_fenetre)

    if touche in (13, ord("a"), ord("A")):
        return detection, "continuer"

    if touche in (ord("m"), ord("M")):
        detection_manuelle = selection_manuelle_tambour(
            image,
            f"Selection manuelle - {nom_image}"
        )
        return detection_manuelle, "continuer"

    if touche in (ord("s"), ord("S")):
        return None, "ignorer"

    if touche in (ord("q"), ord("Q")):
        return None, "quitter"

    return detection, "continuer"


# 9. EXTRACTION, MASQUE NOIR ET CENTRAGE


def extraire_et_centrer_tambour(image, detection):
    """
    Produit une image carrée TAILLE_SORTIE x TAILLE_SORTIE.

    Seuls les pixels appartenant au tambour sont conservés.
    Tous les pixels extérieurs deviennent noirs.
    """

    centre_x = int(detection["centre_x"])
    centre_y = int(detection["centre_y"])
    rayon = int(detection["rayon"])

    hauteur_image, largeur_image = image.shape[:2]

    rayon = max(10, rayon)

    rayon_recadrage = int(
        round(rayon * (1.0 + MARGE_AUTOUR_PIECE))
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
    destination_x2 = destination_x1 + (source_x2 - source_x1)
    destination_y2 = destination_y1 + (source_y2 - source_y1)

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

    centre_carre = (rayon_recadrage, rayon_recadrage)

    cv2.circle(
        masque,
        centre_carre,
        rayon,
        255,
        thickness=-1,
        lineType=cv2.LINE_AA
    )

    piece_sur_fond_noir = cv2.bitwise_and(
        carre_noir,
        carre_noir,
        mask=masque
    )

    image_finale = cv2.resize(
        piece_sur_fond_noir,
        (TAILLE_SORTIE, TAILLE_SORTIE),
        interpolation=cv2.INTER_AREA
    )

    diametre_final_pixels = int(
        round(
            TAILLE_SORTIE
            * (2.0 * rayon)
            / float(taille_carre)
        )
    )

    return image_finale, diametre_final_pixels


# 10. CRÉATION DE L'IMAGE DE DIAGNOSTIC

def creer_image_diagnostic(image, detection, nom_image):
    """
    Crée une copie de l'image originale avec le cercle de détection.
    """

    diagnostic = image.copy()

    centre = (
        int(detection["centre_x"]),
        int(detection["centre_y"])
    )

    rayon = int(detection["rayon"])

    cv2.circle(
        diagnostic,
        centre,
        rayon,
        (0, 255, 0),
        8
    )

    cv2.circle(
        diagnostic,
        centre,
        15,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        diagnostic,
        f"{nom_image} | centre={centre} | rayon={rayon}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3
    )

    return diagnostic


# 11. TRAITEMENT D'UNE SEULE IMAGE

def traiter_une_image(chemin_entree, dossier_sortie, nom_face):
    """
    Traite une image et retourne un dictionnaire pour le rapport CSV.
    """

    nom_image = chemin_entree.name

    rapport = {
        "date_heure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "face": nom_face,
        "image_entree": nom_image,
        "statut": "ERREUR",
        "methode_detection": "",
        "centre_x_original": "",
        "centre_y_original": "",
        "rayon_original": "",
        "diametre_final_px": "",
        "image_sortie": "",
        "message": ""
    }

    try:
        image = lire_image_unicode(chemin_entree)

        detection = detecter_tambour_hough(image)

        detection, action = valider_detection(
            image,
            detection,
            nom_image
        )

        if action == "quitter":
            rapport["statut"] = "ARRET"
            rapport["message"] = "Arrêt demandé par l'utilisateur"
            return rapport, "quitter"

        if action == "ignorer" or detection is None:
            rapport["statut"] = "IGNORE"
            rapport["message"] = "Image ignorée ou sélection annulée"
            return rapport, "continuer"

        image_finale, diametre_final_px = extraire_et_centrer_tambour(
            image,
            detection
        )

        # Sauvegarde en PNG pour éviter les artefacts autour du masque.
        nom_sortie = chemin_entree.stem + "_CENTREE.png"

        chemin_sortie = dossier_sortie / nom_sortie

        sauvegarder_image_unicode(
            chemin_sortie,
            image_finale
        )

        diagnostic = creer_image_diagnostic(
            image,
            detection,
            nom_image
        )

        chemin_diagnostic = (
            DOSSIER_DIAGNOSTIC
            / f"{nom_face}_{chemin_entree.stem}_diagnostic.jpg"
        )

        sauvegarder_image_unicode(
            chemin_diagnostic,
            diagnostic
        )

        rapport["statut"] = "OK"
        rapport["methode_detection"] = detection["methode"]
        rapport["centre_x_original"] = detection["centre_x"]
        rapport["centre_y_original"] = detection["centre_y"]
        rapport["rayon_original"] = detection["rayon"]
        rapport["diametre_final_px"] = diametre_final_px
        rapport["image_sortie"] = chemin_sortie.name
        rapport["message"] = "Image centrée et fond extérieur remplacé par du noir"

        print(
            f"[OK] {nom_image} -> {chemin_sortie.name}"
        )

        return rapport, "continuer"

    except Exception as erreur:
        # Une erreur sur une image ne bloque pas le traitement des suivantes.
        rapport["statut"] = "ERREUR"
        rapport["message"] = str(erreur)

        print(
            f"[ERREUR] {nom_image} : {erreur}"
        )

        return rapport, "continuer"


# 12. TRAITEMENT COMPLET D'UN DOSSIER

def traiter_dossier(dossier_entree, dossier_sortie, nom_face):
    """
    Traite toutes les images d'une face.
    """

    images = lister_images(dossier_entree)

    print(
        f"\n=== {nom_face} : {len(images)} image(s) trouvée(s) ==="
    )

    rapports = []

    for numero, chemin_image in enumerate(images, start=1):

        print(
            f"\n[{numero}/{len(images)}] Traitement de {chemin_image.name}"
        )

        rapport, action = traiter_une_image(
            chemin_image,
            dossier_sortie,
            nom_face
        )

        rapports.append(rapport)

        if action == "quitter":
            break

    return rapports


# 13. SAUVEGARDE DU RAPPORT CSV

def sauvegarder_rapport_csv(rapports):
    """
    Sauvegarde le rapport de traitement dans un fichier CSV.
    """

    colonnes = [
        "date_heure",
        "face",
        "image_entree",
        "statut",
        "methode_detection",
        "centre_x_original",
        "centre_y_original",
        "rayon_original",
        "diametre_final_px",
        "image_sortie",
        "message"
    ]

    with open(
        FICHIER_RAPPORT,
        mode="w",
        newline="",
        encoding="utf-8-sig"
    ) as fichier:

        writer = csv.DictWriter(
            fichier,
            fieldnames=colonnes,
            delimiter=";"
        )

        writer.writeheader()

        for rapport in rapports:
            writer.writerow(rapport)

    print(
        f"\nRapport sauvegardé : {FICHIER_RAPPORT}"
    )


# 14. PROGRAMME PRINCIPAL


def main():
    """
    Lance le traitement de la face avant puis de la face arrière.
    """

    print("============================================================")
    print(" PRÉTRAITEMENT DE LA BASE D'IMAGES — TAMBOURS MTV9")
    print("============================================================")

    print(f"Dossier principal       : {DOSSIER_BASE}")
    print(f"Entrée face avant       : {DOSSIER_ENTREE_AVANT}")
    print(f"Entrée face arrière     : {DOSSIER_ENTREE_ARRIERE}")
    print(f"Sortie face avant       : {DOSSIER_SORTIE_AVANT}")
    print(f"Sortie face arrière     : {DOSSIER_SORTIE_ARRIERE}")
    print(f"Taille finale           : {TAILLE_SORTIE} x {TAILLE_SORTIE}")
    print(f"Validation interactive  : {VALIDATION_INTERACTIVE}")

    tous_les_rapports = []

    try:
        rapports_avant = traiter_dossier(
            DOSSIER_ENTREE_AVANT,
            DOSSIER_SORTIE_AVANT,
            "Face_Avant"
        )

        tous_les_rapports.extend(rapports_avant)

        arret_demande = any(
            rapport["statut"] == "ARRET"
            for rapport in rapports_avant
        )

        if not arret_demande:

            rapports_arriere = traiter_dossier(
                DOSSIER_ENTREE_ARRIERE,
                DOSSIER_SORTIE_ARRIERE,
                "Face_Arriere"
            )

            tous_les_rapports.extend(rapports_arriere)

        sauvegarder_rapport_csv(tous_les_rapports)

        nombre_ok = sum(
            rapport["statut"] == "OK"
            for rapport in tous_les_rapports
        )

        nombre_ignore = sum(
            rapport["statut"] == "IGNORE"
            for rapport in tous_les_rapports
        )

        nombre_erreur = sum(
            rapport["statut"] == "ERREUR"
            for rapport in tous_les_rapports
        )

        print("\n============================================================")
        print(" TRAITEMENT TERMINÉ")
        print("============================================================")
        print(f"Images traitées avec succès : {nombre_ok}")
        print(f"Images ignorées             : {nombre_ignore}")
        print(f"Erreurs                     : {nombre_erreur}")
        print(f"Résultats avant             : {DOSSIER_SORTIE_AVANT}")
        print(f"Résultats arrière           : {DOSSIER_SORTIE_ARRIERE}")
        print(f"Diagnostics                 : {DOSSIER_DIAGNOSTIC}")
        print(f"Rapport                     : {FICHIER_RAPPORT}")

    finally:
        # Fermeture des fenêtres OpenCV en fin de traitement.
        cv2.destroyAllWindows()


# Point d'entrée du programme.
if __name__ == "__main__":
    main()

\# Source des données



\## Provenance



\- \*\*Dataset\*\* : IBM Telco Customer Churn

\- \*\*Origine\*\* : copie fournie par l'enseignant

\- \*\*Emplacement local\*\* : `data/raw/telco.csv`

\- \*\*Statut\*\* : fichier original, non modifié (ne pas l'ouvrir/réenregistrer dans un tableur)

\- \*\*SHA-256\*\* : `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

Le fichier reçu initialement avait toutes ses lignes entourées de guillemets (réexport probable). Il a été remplacé par la version officielle IBM commit-pinnée (commit 1fd6fd7...). L’ancien fichier est conservé localement sous telco_quoted_received.csv.


\## Vérifier le fichier



Depuis la racine du projet (Bash) :



```bash

sha256sum data/raw/telco.csv

```



Le résultat doit être identique au hash ci-dessus. Sinon, ce n'est pas le même fichier.



Le CSV n'est pas versionné dans Git (voir `.gitignore`) : il faut le récupérer auprès de l'enseignant ou via le lien commit-pinné du sujet, puis le placer dans `data/raw/telco.csv`.



\## Description des colonnes



Il s'agit d'un instantané clients (une ligne par client), pas d'un historique daté.



| Rôle | Colonnes |

|---|---|

| Identifiant client | `customerID` |

| Cible | `Churn` (valeurs `Yes` / `No`) |

| Numériques | `tenure`, `MonthlyCharges`, `TotalCharges` |

| Catégorielles | `gender`, `SeniorCitizen` (codée 0/1), `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |



\## Remarques



\- `customerID` et `Churn` ne doivent pas devenir des entrées ordinaires du modèle (identifiant sans signal généralisable, et cible à prédire).

\- `TotalCharges` peut être lue comme du texte si certaines valeurs sont vides ou composées d'espaces : à vérifier avec le rapport d'inspection (`reports/data\_overview.json`).

\- Un hash identique prouve que le fichier est identique, pas que les données métier sont correctes.


![](https://avatars1.githubusercontent.com/u/63645182?s=200&v=4)

# Tableau de bord de l'apprentissage - Laboratoire

## Fiche Produit

## Documentation

## 1. Test application
### Install requirements
```shell
$ cd server && python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
$ python -m spacy download fr_core_news_lg
```

### Running development server
```shell
$ python main.py
```

### Test endpoint
#### Check API status
```shell
$ curl http://127.0.0.1:8000/

{"status":"TBA classifier API ready."}
```
#### Load model version
```shell
curl http://127.0.0.1:8000/model/load?version='2025-10-20'

{"model":"2025-10-20"}
```

#### Check model version
```shell
$ curl http://127.0.0.1:8000/model/version

{"model":"2025-10-20"}
```
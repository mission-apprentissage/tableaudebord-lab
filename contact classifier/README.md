## 1. Test application

### Quick Start (npm-style commands)

```shell
make help          # Show available commands
make install       # Install Python dependencies locally
make dev           # Run development server locally with hot-reload
make dev-up        # Start with Docker Compose (development with hot-reload)
make down          # Stop all services
make test          # Test API endpoints
```

### Install requirements

```shell
make install
```

### Running development server

**Option 1 : Local (recommandé pour développement)**

```shell
$ make dev
```

Lance le serveur en local avec hot-reload activé. Les modifications de code rechargent automatiquement le serveur.

**Option 2 : Docker**

```shell
$ make dev-up
```

Lance le serveur dans Docker avec hot-reload via volume monté.

**Note importante** : Le serveur charge automatiquement le dernier modèle disponible sur HuggingFace au démarrage. Si aucun modèle n'est disponible, le serveur ne démarre pas et affiche un message d'erreur clair dans les logs.

### Test endpoints

#### Check API status

```shell
$ curl http://127.0.0.1:8000/

{"status":"TBA classifier API ready."}
```

#### Check model version (auto-loaded at startup)

```shell
$ curl http://127.0.0.1:8000/model/version

{"model":"2026-03-16"}
```

#### Load specific model version

```shell
$ curl http://127.0.0.1:8000/model/load?version=2026-03-16

{"model":"2026-03-16"}
```

#### Score texts (without version - uses auto-loaded model)
```shell
$ curl http://127.0.0.1:8000/model/score \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "data": [
      {
        "apprenant.date_de_naissance": "2002-07-28T00:00:00.000Z",
        "formation.date_inscription": "2025-11-10T00:00:00.000Z",
        "formation.date_fin": "2027-05-09T00:00:00.000Z",
        "formation.date_entree": "2025-11-10T00:00:00.000Z",
        "contrat.date_debut": "2025-11-10T00:00:00.000Z",
        "contrat.date_fin": "2027-05-09T00:00:00.000Z",
        "contrat.date_rupture": "2025-12-15T00:00:00.000Z"
      },
      {
        "apprenant.date_de_naissance": "2001-04-07T00:00:00.000Z",
        "formation.date_inscription": "2024-12-05T00:00:00.000Z",
        "formation.date_fin": "2026-10-01T00:00:00.000Z",
        "formation.date_entree": "2024-12-05T00:00:00.000Z",
        "contrat.date_debut": "2024-12-02T00:00:00.000Z",
        "contrat.date_fin": "2026-10-01T00:00:00.000Z",
        "contrat.date_rupture": "2026-02-03T00:00:00.000Z"
      },
      {
        "apprenant.date_de_naissance": "2007-08-12T00:00:00.000Z",
        "formation.date_inscription": "2024-04-10T11:29:00.000Z",
        "formation.date_fin": "2025-06-30T00:00:00.000Z",
        "formation.date_entree": "2024-09-03T00:00:00.000Z",
        "contrat.date_debut": "2023-09-01T00:00:00.000Z",
        "contrat.date_fin": "2025-08-31T00:00:00.000Z",
        "contrat.date_rupture": "2024-12-31T00:00:00.000Z"
      }
    ]
  }'

{
  "model": "2026-03-16",
  "scores": [
    0.51,
    0.91,
    0.14
  ]
}
```
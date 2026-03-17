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

{"model":"2025-12-18"}
```
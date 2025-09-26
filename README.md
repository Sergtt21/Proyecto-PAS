# Proyecto-PAS
Proyecto realizado para la materia de progracion avanzada de sistemas inteligentes, aplicando OpenCV.

## Prueba de integración (bus + bot)

Este proyecto incluye un script llamado `send_test_event.py` que sirve para comprobar que la cola de eventos (`bus.py`) y el bot de Telegram (`bot.py`) se comunican correctamente.

### Prerequisitos
1. Tener **Redis** en funcionamiento.  
   Ejemplo rápido con Docker:
   ```bash
   docker run -d --name redis -p 6379:6379 redis:7

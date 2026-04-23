# Guía de Contribución

¡Gracias por tu interés en mejorar `face-match`! 

## Cómo contribuir

1. **Reportar Bugs**: Abre un *Issue* describiendo el problema, tu sistema operativo y cómo reproducirlo.
2. **Sugerir Mejoras**: Si tienes una idea para una nueva funcionalidad, abre un *Issue* para discutirla.
3. **Enviar Código**:
    - Haz un *Fork* del repositorio.
    - Crea una rama para tu mejora (`git checkout -b feature/mejora`).
    - Asegúrate de que los tests pasen (`pytest`).
    - Envía un *Pull Request*.

## Estándares de Código

- Usamos **anotaciones de tipos** (type hints) en todas las funciones nuevas.
- El código debe seguir el estilo **PEP 8**.
- Si añades una funcionalidad, añade un test en la carpeta `tests/`.

## Reporte de Seguridad

Si encuentras una vulnerabilidad de seguridad, por favor no abras un issue público. Envía un correo a los mantenedores para resolverlo de forma privada.

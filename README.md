# Análisis histórico de acciones (2010–2024)

Proyecto de análisis bursátil desarrollado en **Python** para explorar el comportamiento histórico de acciones (aprox. 15 años).
Incluye visualización de **precios y volumen**, análisis exploratorio y (según la versión) **indicadores técnicos** y análisis de patrones con **FFT**.

---

## Archivos

- `Datos bursátiles de 15 años de NVDA AAPL M...` — Dataset / archivo principal con datos históricos.
- `proyecto_off.py` — Script principal del proyecto (ejecución local/offline).
- `proyecto_sintalib.py` — Versión alternativa **sin TA-Lib** (útil si no puedes instalar TA-Lib).
- `requisitos.txt` — Dependencias del proyecto.
- `README.md` — Este documento.

---

## Requisitos

- **Local:** Python 3.9+ (recomendado) y pip.
- **Nota:** si tienes problemas instalando **TA-Lib**, utiliza `proyecto_sintalib.py`.

---

## Instalación (Python)

```bash
pip install -r requisitos.txt

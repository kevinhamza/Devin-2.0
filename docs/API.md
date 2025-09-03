# Devin AGI - Backend Server API Documentation

This document provides details on the primary REST API endpoints for Devin's backend services.

---

## 1. Analytics Server

**Base URL:** `http://localhost:5004`

### Log an Event
- **Endpoint:** `/log`
- **Method:** `POST`
- **Description:** Logs a single, timestamped event to the in-memory analytics database.
- **Request Body (JSON):**
  ```json
  {
    "event_type": "cpu_usage",
    "value": 75.5
  }

import scripts.smoke_test_api as smoke_test


def test_wait_for_health_retries_connection_reset(monkeypatch):
    responses = iter([ConnectionResetError("still starting"), {"status": "ok"}])

    def fake_get_json(_url: str) -> dict:
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(smoke_test, "_get_json", fake_get_json)
    monkeypatch.setattr(smoke_test.time, "sleep", lambda _seconds: None)

    smoke_test.wait_for_health("http://127.0.0.1:8000", timeout=1.0)

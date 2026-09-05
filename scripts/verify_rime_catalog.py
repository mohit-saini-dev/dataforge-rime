import json
import urllib.request


CATALOG_URL = "https://users.rime.ai/data/voices/all-v2.json"


def main():
    print("Fetching Rime voice catalog...")

    with urllib.request.urlopen(CATALOG_URL, timeout=10) as response:
        catalog = json.load(response)

    print("\nAvailable Rime models and voices:\n")

    for model, languages in catalog.items():
        print(f"Model: {model}")

        for language, voices in languages.items():
            print(f"  {language}: {len(voices)} voices")

    print("\nRime catalog verification: OK")


if __name__ == "__main__":
    main()
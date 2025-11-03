from dotenv import load_dotenv
import os

print("=" * 60)
print("🔍 Testing .env Configuration")
print("=" * 60)

# Load .env file
load_dotenv()

# Check each variable
print("\n📋 Configuration Values:\n")

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    print(f"✅ GEMINI_API_KEY: {gemini_key[:10]}...{gemini_key[-5:]}")
    print(f"   Length: {len(gemini_key)} characters")
    print(f"   Format: {'✅ Correct' if gemini_key.startswith('AIza') else '❌ Incorrect'}")
else:
    print("❌ GEMINI_API_KEY: Not found")

email = os.getenv("ENTREZ_EMAIL")
if email:
    print(f"✅ ENTREZ_EMAIL: {email}")
else:
    print("❌ ENTREZ_EMAIL: Not found")

max_results = os.getenv("MAX_RESULTS")
if max_results:
    print(f"✅ MAX_RESULTS: {max_results}")
else:
    print("❌ MAX_RESULTS: Not found (will use default: 10)")

print("\n" + "=" * 60)

if gemini_key and email:
    print("✅ Configuration is complete!")
    print("   You're ready to run the Streamlit app!")
else:
    print("❌ Configuration incomplete - check your .env file")

print("=" * 60)
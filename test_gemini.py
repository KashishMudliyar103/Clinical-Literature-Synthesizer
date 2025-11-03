"""
Advanced Gemini API Key Test Script
Tests all possible model names and API versions
"""

import google.generativeai as genai
import sys

print("=" * 70)
print("🔬 ADVANCED GEMINI API KEY TESTER")
print("=" * 70)

# ============================================
# STEP 1: Your API Key
# ============================================
API_KEY = "AIzaSyBKGHFMUDcqQqw2ax-zFZwNj3peIFgI7f4"

print(f"\n🔍 Testing API key: {API_KEY[:10]}...{API_KEY[-5:]}")
print(f"   Length: {len(API_KEY)} characters")
print(f"   Format: {'✅ Valid' if API_KEY.startswith('AIza') else '❌ Invalid'}")

# ============================================
# STEP 2: Configure API
# ============================================
print("\n⏳ Configuring Gemini API...")

try:
    genai.configure(api_key=API_KEY)
    print("✅ API configured")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    sys.exit(1)

# ============================================
# STEP 3: List ALL Available Models
# ============================================
print("\n" + "=" * 70)
print("🔍 DISCOVERING AVAILABLE MODELS")
print("=" * 70)

try:
    print("\n⏳ Fetching your account's available models...")
    
    models_list = list(genai.list_models())
    
    if not models_list:
        print("\n❌ No models found! This means:")
        print("   • Your API key might not be activated yet")
        print("   • Wait 2-5 minutes and try again")
        print("   • Or create a completely new API key")
        sys.exit(1)
    
    print(f"\n✅ Found {len(models_list)} total models\n")
    
    # Filter for generation models
    generation_models = []
    
    for model in models_list:
        methods = getattr(model, 'supported_generation_methods', [])
        
        print(f"📦 {model.name}")
        print(f"   Supported methods: {', '.join(methods) if methods else 'None'}")
        
        if 'generateContent' in methods:
            generation_models.append(model.name)
            print(f"   ✅ Can generate content")
        else:
            print(f"   ⚠️  Cannot generate content")
        print()
    
    if not generation_models:
        print("\n❌ No models support content generation!")
        print("   Your API key might have restrictions.")
        sys.exit(1)
    
    print("=" * 70)
    print(f"✅ {len(generation_models)} models can generate content:")
    for m in generation_models:
        print(f"   • {m}")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Failed to list models: {e}")
    print("\n🔧 This usually means:")
    print("   1. API key is invalid")
    print("   2. API key needs more time to activate (wait 5 mins)")
    print("   3. Network/firewall blocking the request")
    print("   4. Regional restrictions")
    sys.exit(1)

# ============================================
# STEP 4: Test Content Generation
# ============================================
print("\n" + "=" * 70)
print("🧪 TESTING CONTENT GENERATION")
print("=" * 70)

working_model = None
response = None

for model_name in generation_models:
    try:
        print(f"\n⏳ Testing: {model_name}...", end=" ")
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'Hello World' in one sentence")
        
        if response and response.text:
            print("✅ SUCCESS!")
            working_model = model_name
            print(f"   Response: {response.text[:100]}")
            break
        else:
            print("⚠️  No response text")
            
    except Exception as e:
        print(f"❌ Failed")
        print(f"   Error: {str(e)[:80]}")

# ============================================
# STEP 5: Final Results
# ============================================
print("\n" + "=" * 70)
print("📊 FINAL RESULTS")
print("=" * 70)

if working_model and response:
    print("\n🎉 SUCCESS! Everything is working!\n")
    print(f"✅ Your API Key: {API_KEY}")
    print(f"✅ Working Model: {working_model}")
    print(f"\n🤖 Test Response:")
    print(f"   {response.text}\n")
    
    print("=" * 70)
    print("📝 NEXT STEPS:")
    print("=" * 70)
    print("\n1. Update your .env file:")
    print(f"   GEMINI_API_KEY={API_KEY}")
    print(f"\n2. Update app.py to use this model:")
    print(f"   model = genai.GenerativeModel('{working_model}')")
    print("\n3. Restart Streamlit:")
    print("   streamlit run app.py")
    print("\n" + "=" * 70)
    
else:
    print("\n❌ NO WORKING MODELS FOUND\n")
    print("=" * 70)
    print("🔧 TROUBLESHOOTING GUIDE")
    print("=" * 70)
    
    print("\n📍 OPTION 1: Create a COMPLETELY NEW API Key")
    print("   1. Go to: https://aistudio.google.com/app/apikey")
    print("   2. DELETE all existing keys")
    print("   3. Click 'Create API Key'")
    print("   4. Choose 'Create API key in new project' ⚠️ IMPORTANT!")
    print("   5. Copy the new key")
    print("   6. Wait 5 minutes (keys need activation time)")
    print("   7. Run this test again")
    
    print("\n📍 OPTION 2: Check Your Google Account")
    print("   • Make sure you're using a personal Google account")
    print("   • Not a workspace/organization account")
    print("   • Sign out and sign back in")
    
    print("\n📍 OPTION 3: Regional Issues")
    print("   • Gemini might not be available in your country")
    print("   • Try using a VPN to a supported region")
    print("   • Supported: US, UK, EU countries")
    
    print("\n📍 OPTION 4: Use PubMed-Only Version")
    print("   • Your project works fine without Gemini!")
    print("   • PubMed search is working (9 studies found)")
    print("   • Use the enhanced version without AI")
    
    print("\n" + "=" * 70)

print("\n💾 Debug Info:")
print(f"   Python version: {sys.version.split()[0]}")
print(f"   genai library: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")
print("=" * 70)
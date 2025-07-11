# ---------------- PRICE-BASED CROP RECOMMENDATION ----------------
st.subheader("💰 Price-Based Crop Recommendation")

user_price = st.number_input("Enter your expected crop price (₹ per quintal)", min_value=0)

def get_crop_prices():
    url = "https://api.data.gov.in/resource/f9efb243-4f43-4941-a181-0a6e54c5f295"
    params = {
        "api-key": "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b",
        "format": "json",
        "limit": 100
    }
    try:
        res = requests.get(url, params=params)
        data = res.json()
        prices = []
        for entry in data['records']:
            try:
                modal_price = int(entry['modal_price'])
                crop = entry['commodity']
                if modal_price >= user_price:
                    prices.append((crop, modal_price, entry['state'], entry['market']))
            except:
                continue
        return prices
    except:
        return []

if st.button("Suggest Crops by Price"):
    if user_price <= 0:
        st.warning("Please enter a valid price.")
    else:
        matching_crops = get_crop_prices()
        if matching_crops:
            st.success("🌾 Crops with prices at or above your input:")
            for crop, price, state, market in matching_crops[:10]:  # Show top 10
                st.write(f"**{crop}** – ₹{price} (State: {state}, Market: {market})")
        else:
            st.warning("No crops found with modal prices above the given value.")

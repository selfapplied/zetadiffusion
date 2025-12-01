#!.venv/bin/python
"""
Cryptocurrency Investment Predictions using Field Equations

Models cryptocurrency markets as fields and predicts:
- Price movements
- Volatility patterns
- Market cycles
- Investment opportunities

Uses the universal field equation framework to analyze crypto data.
"""

import sys
import numpy as np
from pathlib import Path
from zetadiffusion.compress import compress_text, encode_text_to_state
from zetadiffusion.windowspec import SeedSpec
from zetadiffusion.guardian import SystemState
import json
from datetime import datetime, timedelta

# Cryptocurrency market data (example patterns)
CRYPTO_DATA = {
    "bitcoin": """
    Bitcoin (BTC) market dynamics:
    Price equation: dP/dt = μP(1 - P/K) + σP dW (logistic growth with noise)
    Volatility: σ = σ₀ * exp(λt) (exponential volatility)
    Market cap: M = P * S (price × supply)
    Hash rate: H(t) = H₀ * (1 + r)^t (exponential growth)
    Network effect: N(t) = N₀ / (1 + exp(-k(t-t₀))) (S-curve adoption)
    """,
    
    "ethereum": """
    Ethereum (ETH) market dynamics:
    Price equation: dP/dt = αP + βP² + σP dW (quadratic growth)
    Gas price: G = G₀ * (1 + demand/supply) (supply-demand)
    Staking yield: Y = (total_staked / total_supply) * reward_rate
    DeFi TVL: TVL(t) = TVL₀ * exp(rt) (exponential growth)
    Network activity: A = transactions_per_block * block_time
    """,
    
    "defi": """
    DeFi (Decentralized Finance) market dynamics:
    Total Value Locked: TVL(t) = Σ protocols (liquidity pools)
    Yield farming: APY = rewards / staked * 365
    Impermanent loss: IL = 2√(price_ratio) / (1 + price_ratio) - 1
    Liquidity provision: LP_return = fees + rewards - IL
    Protocol revenue: R = fees * volume * protocol_fee_rate
    """,
    
    "nft": """
    NFT (Non-Fungible Token) market dynamics:
    Floor price: F(t) = F₀ * (1 + trend) + volatility * noise
    Trading volume: V(t) = V₀ * exp(-decay * t) + spikes
    Rarity score: R = Σ traits (1 / frequency_trait)
    Market sentiment: S = (buys - sells) / (buys + sells)
    Collection value: C = floor_price * supply * premium
    """,
    
    "stablecoins": """
    Stablecoin market dynamics:
    Peg maintenance: P = target ± band (price band)
    Collateral ratio: CR = collateral_value / stablecoin_supply
    Arbitrage: A = (market_price - peg) / transaction_cost
    Reserve backing: R = fiat_reserves / total_supply
    Depegging risk: Risk = |P - peg| / volatility
    """
}

def predict_price_movement(state: SystemState, current_price: float, time_horizon: int = 30) -> dict:
    """
    Predict price movement using operator spectrum.
    
    Field equation: dP/dt = D∇²P + f(P) + η(t)
    Where:
    - D (diffusion) from coherence (market structure)
    - f(P) from chaos (volatility)
    - Gradients from stress (market pressure)
    - Memory from hurst (trend persistence)
    """
    coherence = state.coherence
    chaos = state.chaos
    stress = state.stress
    hurst = state.hurst
    
    # Price evolution model
    # dP/dt = μP + σP dW
    # μ (drift) from coherence (structure)
    # σ (volatility) from chaos (complexity)
    
    drift = coherence * 0.01  # Positive drift if structured
    volatility = chaos * 0.05  # Volatility from chaos
    
    # Predict price path
    prices = [current_price]
    for t in range(1, time_horizon + 1):
        # Geometric Brownian Motion with Hurst memory
        if hurst > 0.5:
            # Persistent (trend-following)
            trend = drift * (1 + hurst - 0.5)
        else:
            # Anti-persistent (mean-reverting)
            trend = drift * (1 - (0.5 - hurst))
        
        # Random walk component
        noise = np.random.normal(0, volatility)
        
        # Update price
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(max(0.01, new_price))  # Floor at 0.01
    
    # Calculate predictions
    final_price = prices[-1]
    price_change = final_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Volatility estimate
    returns = np.diff(prices) / prices[:-1]
    predicted_volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized %
    
    # Trend direction
    if price_change > 0:
        trend_direction = "BULLISH"
        confidence = min(0.95, coherence + (price_change_pct / 100))
    else:
        trend_direction = "BEARISH"
        confidence = min(0.95, coherence + abs(price_change_pct / 100))
    
    return {
        'current_price': current_price,
        'predicted_price': final_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'trend_direction': trend_direction,
        'confidence': confidence,
        'predicted_volatility': predicted_volatility,
        'price_path': prices,
        'time_horizon_days': time_horizon,
        'operator_spectrum': {
            'coherence': coherence,
            'chaos': chaos,
            'stress': stress,
            'hurst': hurst
        }
    }

def generate_investment_recommendation(prediction: dict, risk_tolerance: str = "moderate") -> dict:
    """Generate investment recommendation from prediction."""
    
    price_change_pct = prediction['price_change_pct']
    confidence = prediction['confidence']
    volatility = prediction['predicted_volatility']
    
    # Risk assessment
    if volatility > 100:
        risk_level = "HIGH"
    elif volatility > 50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # Recommendation logic
    if price_change_pct > 10 and confidence > 0.7:
        action = "STRONG BUY"
        allocation = 0.15 if risk_tolerance == "aggressive" else 0.10
    elif price_change_pct > 5 and confidence > 0.6:
        action = "BUY"
        allocation = 0.10 if risk_tolerance == "aggressive" else 0.05
    elif price_change_pct > 0 and confidence > 0.5:
        action = "HOLD"
        allocation = 0.05
    elif price_change_pct < -10 and confidence > 0.7:
        action = "STRONG SELL"
        allocation = 0.0
    elif price_change_pct < -5 and confidence > 0.6:
        action = "SELL"
        allocation = 0.0
    else:
        action = "HOLD"
        allocation = 0.02
    
    # Adjust for risk tolerance
    if risk_tolerance == "conservative":
        allocation *= 0.5
    elif risk_tolerance == "aggressive":
        allocation *= 1.5
        allocation = min(0.20, allocation)  # Cap at 20%
    
    return {
        'action': action,
        'recommended_allocation': allocation,
        'risk_level': risk_level,
        'expected_return_pct': price_change_pct,
        'confidence': confidence,
        'time_horizon_days': prediction['time_horizon_days']
    }

def showcase_crypto_predictions():
    """Showcase cryptocurrency investment predictions."""
    
    print("=" * 70)
    print("CRYPTOCURRENCY INVESTMENT PREDICTIONS")
    print("=" * 70)
    print()
    print("Using Universal Field Equations to predict:")
    print("  - Price movements")
    print("  - Volatility patterns")
    print("  - Market cycles")
    print("  - Investment opportunities")
    print()
    
    Path(".out").mkdir(exist_ok=True)
    
    # Example current prices (in USD)
    # Can be overridden from command line
    current_prices = {
        "bitcoin": 45000.0,
        "ethereum": 2800.0,
        "defi": 100.0,  # DeFi index
        "nft": 50.0,    # NFT index
        "stablecoins": 1.0  # Stablecoin peg
    }
    
    # Parse command line for custom prices
    if len(sys.argv) > 1:
        try:
            # Format: python demo_crypto_predictions.py bitcoin=50000 ethereum=3000
            for arg in sys.argv[1:]:
                if '=' in arg:
                    asset, price = arg.split('=')
                    current_prices[asset.lower()] = float(price)
        except:
            pass
    
    all_predictions = {}
    all_recommendations = {}
    
    # Analyze each cryptocurrency
    for crypto, text in CRYPTO_DATA.items():
        print(f"Analyzing: {crypto.upper()}")
        print("-" * 70)
        
        # Compress market data to get operator spectrum
        result = compress_text(text)
        spec = result['spec']
        state = result['state']
        
        # Get current price
        current_price = current_prices.get(crypto, 100.0)
        
        # Predict price movement (30-day horizon)
        prediction = predict_price_movement(state, current_price, time_horizon=30)
        all_predictions[crypto] = prediction
        
        # Generate investment recommendation
        recommendation = generate_investment_recommendation(prediction, risk_tolerance="moderate")
        all_recommendations[crypto] = recommendation
        
        print(f"Current Price: ${current_price:,.2f}")
        print(f"Predicted Price (30 days): ${prediction['predicted_price']:,.2f}")
        print(f"Expected Change: {prediction['price_change_pct']:+.2f}%")
        print(f"Trend: {prediction['trend_direction']}")
        print(f"Confidence: {prediction['confidence']:.1%}")
        print(f"Predicted Volatility: {prediction['predicted_volatility']:.1f}% (annualized)")
        print()
        print(f"Investment Recommendation:")
        print(f"  Action: {recommendation['action']}")
        print(f"  Recommended Allocation: {recommendation['recommended_allocation']:.1%}")
        print(f"  Risk Level: {recommendation['risk_level']}")
        print()
        
        # Save compressed signature
        spec.to_file(f"{crypto}_compressed.json")
    
    # Portfolio summary
    print("=" * 70)
    print("PORTFOLIO SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Asset':<15} | {'Action':<12} | {'Allocation':<12} | {'Expected Return':<15} | {'Risk':<8}")
    print("-" * 70)
    
    total_allocation = 0.0
    weighted_return = 0.0
    
    for crypto, rec in all_recommendations.items():
        pred = all_predictions[crypto]
        total_allocation += rec['recommended_allocation']
        weighted_return += rec['recommended_allocation'] * pred['price_change_pct']
        print(f"{crypto.upper():<15} | {rec['action']:<12} | {rec['recommended_allocation']:>11.1%} | "
              f"{pred['price_change_pct']:>+14.2f}% | {rec['risk_level']:<8}")
    
    print("-" * 70)
    print(f"{'TOTAL':<15} | {'':<12} | {total_allocation:>11.1%} | {weighted_return:>+14.2f}% | {'':<8}")
    print()
    
    # High-value investments
    print("=" * 70)
    print("HIGH-VALUE INVESTMENT OPPORTUNITIES")
    print("=" * 70)
    print()
    
    # Sort by expected return
    sorted_predictions = sorted(all_predictions.items(), 
                               key=lambda x: x[1]['price_change_pct'], 
                               reverse=True)
    
    print("Ranked by Expected Return:")
    print("-" * 70)
    print(f"{'Rank':<6} | {'Asset':<15} | {'Return':<12} | {'Confidence':<12} | {'Risk':<8} | {'Action':<12}")
    print("-" * 70)
    
    high_value = []
    for i, (crypto, pred) in enumerate(sorted_predictions, 1):
        rec = all_recommendations[crypto]
        print(f"{i:<6} | {crypto.upper():<15} | {pred['price_change_pct']:>+11.2f}% | "
              f"{pred['confidence']:>11.1%} | {rec['risk_level']:<8} | {rec['action']:<12}")
        
        # High-value criteria: >15% return, >60% confidence, LOW/MEDIUM risk
        if pred['price_change_pct'] > 15 and pred['confidence'] > 0.6 and rec['risk_level'] in ['LOW', 'MEDIUM']:
            high_value.append((crypto, pred, rec))
    
    print()
    
    if high_value:
        print("⭐ HIGH-VALUE INVESTMENTS (Meet all criteria):")
        print("-" * 70)
        for crypto, pred, rec in high_value:
            print(f"{crypto.upper()}:")
            print(f"  Expected Return: {pred['price_change_pct']:+.2f}%")
            print(f"  Confidence: {pred['confidence']:.1%}")
            print(f"  Risk: {rec['risk_level']}")
            print(f"  Action: {rec['action']} ({rec['recommended_allocation']:.1%} allocation)")
            print(f"  Current: ${pred['current_price']:,.2f} → Predicted: ${pred['predicted_price']:,.2f}")
            print()
    else:
        print("Top Opportunities:")
        print("-" * 70)
        top = sorted_predictions[0]
        pred = top[1]
        rec = all_recommendations[top[0]]
        print(f"{top[0].upper()}: {pred['price_change_pct']:+.2f}% return, {pred['confidence']:.1%} confidence")
        print()
    
    # Market outlook
    print("Market Outlook:")
    print("-" * 70)
    
    bullish_count = sum(1 for p in all_predictions.values() if p['trend_direction'] == 'BULLISH')
    bearish_count = sum(1 for p in all_predictions.values() if p['trend_direction'] == 'BEARISH')
    
    if bullish_count > bearish_count:
        outlook = "BULLISH"
        sentiment = "Positive - Most assets expected to appreciate"
    elif bearish_count > bullish_count:
        outlook = "BEARISH"
        sentiment = "Negative - Most assets expected to depreciate"
    else:
        outlook = "NEUTRAL"
        sentiment = "Mixed - Market in transition"
    
    print(f"Overall Market: {outlook}")
    print(f"Sentiment: {sentiment}")
    print(f"Bullish assets: {bullish_count}/{len(all_predictions)}")
    print(f"Bearish assets: {bearish_count}/{len(all_predictions)}")
    print()
    
    # Save results
    results_file = ".out/crypto_predictions.json"
    with open(results_file, 'w') as f:
        json.dump({
            'predictions': {
                crypto: {
                    'current_price': pred['current_price'],
                    'predicted_price': pred['predicted_price'],
                    'price_change_pct': pred['price_change_pct'],
                    'trend_direction': pred['trend_direction'],
                    'confidence': pred['confidence'],
                    'volatility': pred['predicted_volatility'],
                    'operator_spectrum': pred['operator_spectrum']
                }
                for crypto, pred in all_predictions.items()
            },
            'recommendations': all_recommendations,
            'portfolio': {
                'total_allocation': total_allocation,
                'weighted_return': weighted_return,
                'market_outlook': outlook
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"Predictions saved to: {results_file}")
    print()
    
    # Disclaimer
    print("=" * 70)
    print("DISCLAIMER")
    print("=" * 70)
    print()
    print("These predictions are based on field equation modeling and")
    print("operator spectrum analysis. They are NOT financial advice.")
    print()
    print("Key assumptions:")
    print("  - Markets follow field dynamics: dP/dt = D∇²P + f(P) + η(t)")
    print("  - Operator spectrum (C, λ, G, H) encodes market structure")
    print("  - Predictions are probabilistic, not deterministic")
    print("  - Past performance does not guarantee future results")
    print()
    print("Always do your own research and consult financial advisors.")
    print()
    print("=" * 70)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    showcase_crypto_predictions()


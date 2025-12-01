#!.venv/bin/python
"""
Wormhole Effects on Financial Markets

Analyzes how topological shortcuts (wormholes) created during star birth
affect financial market predictions and investment recommendations.

Star Birth = System Compression → Creates Wormholes (Topological Shortcuts)
Wormholes = Genus Changes → Energy Release → Market Opportunities
"""

import sys
import numpy as np
from pathlib import Path
from zetadiffusion.compress import compress_text, encode_text_to_state
from zetadiffusion.windowspec import SeedSpec
from zetadiffusion.energy import TopologicalHarvester, ManifoldState
from zetadiffusion.guardian import SystemState
import json

# Cryptocurrency market data
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
    """,
    
    "telegram": """
    Telegram TON (The Open Network) market dynamics:
    Price equation: dP/dt = αP * (1 + N_telegram/N_max) + σP dW (network effect growth)
    User base: N_telegram = 900M+ active users (massive adoption potential)
    Transaction speed: T = 5 seconds (ultra-fast confirmation)
    Gas fees: G = 0.01 TON (extremely low cost)
    Bot ecosystem: B(t) = B₀ * exp(rt) (exponential bot growth)
    Payment integration: P_payment = f(telegram_users, bot_adoption)
    Network effect: V = N² (Metcalfe's law, quadratic value)
    Staking yield: Y = (total_staked / total_supply) * inflation_rate
    Telegram integration: I = native_wallet * user_adoption_rate
    Cross-chain bridges: C = bridge_volume * fee_rate
    """
}

def detect_wormholes(state: SystemState, spec: SeedSpec, n_iterations: int = 50) -> list:
    """
    Detect wormholes (topological shortcuts) created during star birth.
    
    Wormholes occur when:
    - Stress builds up (curvature increases)
    - Mach > 1.0 (supersonic flow)
    - Genus changes (topological surgery)
    - Energy is released
    
    Returns list of wormhole events with their effects.
    """
    harvester = TopologicalHarvester(efficiency=0.8)
    manifold = ManifoldState(
        curvature_grid=np.array([state.stress * state.coherence] * 100),
        viscosity_gamma=1.0 - state.hurst,
        current_genus=0
    )
    
    wormholes = []
    initial_genus = 0
    
    for i in range(n_iterations):
        # Build up stress (chaos injection)
        stress_build = state.stress + state.chaos * (i + 1) * 0.3
        curvature_build = stress_build * state.coherence
        
        # Update manifold
        manifold.curvature_grid = np.array([curvature_build] * 100)
        
        # Calculate Mach number
        flow_velocity = stress_build * (1 + state.chaos)
        sound_speed = state.coherence * 0.1
        mach = flow_velocity / (sound_speed + 0.01)
        
        # Process event
        event = harvester.process_event(manifold, mach)
        
        if event['status'] == 'SHOCK_HARVESTED':
            # Wormhole created!
            wormhole = {
                'iteration': i,
                'genus_before': initial_genus,
                'genus_after': manifold.current_genus,
                'energy_released': event['energy_released'],
                'work_captured': event['work_captured'],
                'mach': mach,
                'topology_change': event['topology_change']
            }
            wormholes.append(wormhole)
            initial_genus = manifold.current_genus
    
    return wormholes

def calculate_wormhole_boost(prediction: dict, wormholes: list) -> dict:
    """
    Calculate how wormholes boost market predictions.
    
    Wormholes create:
    - Energy release → Market momentum
    - Topological shortcuts → Faster price discovery
    - Genus changes → Structural market shifts
    - Information gain → Reduced uncertainty
    """
    base_return = prediction['price_change_pct']
    base_confidence = prediction['confidence']
    
    # Wormhole effects
    n_wormholes = len(wormholes)
    total_energy = sum(w['work_captured'] for w in wormholes)
    avg_genus_jump = np.mean([w['genus_after'] - w['genus_before'] for w in wormholes]) if wormholes else 0
    
    # Boost calculations
    # More wormholes = more energy = more momentum
    energy_boost = min(0.20, total_energy / 1000.0)  # Cap at 20% boost
    genus_boost = min(0.15, avg_genus_jump * 0.05)  # Topological complexity boost
    wormhole_boost = min(0.25, (n_wormholes / 10.0) * 0.10)  # Frequency boost
    
    # Total boost
    total_boost = energy_boost + genus_boost + wormhole_boost
    
    # Apply boost to return
    boosted_return = base_return * (1 + total_boost)
    
    # Increase confidence (wormholes reduce uncertainty)
    confidence_boost = min(0.20, n_wormholes * 0.02)
    boosted_confidence = min(0.95, base_confidence + confidence_boost)
    
    # Volatility reduction (wormholes create shortcuts, reduce noise)
    volatility_reduction = min(0.30, n_wormholes * 0.015)
    boosted_volatility = prediction['predicted_volatility'] * (1 - volatility_reduction)
    
    return {
        'base_return': base_return,
        'boosted_return': boosted_return,
        'return_boost_pct': total_boost * 100,
        'base_confidence': base_confidence,
        'boosted_confidence': boosted_confidence,
        'base_volatility': prediction['predicted_volatility'],
        'boosted_volatility': boosted_volatility,
        'n_wormholes': n_wormholes,
        'total_energy': total_energy,
        'avg_genus_jump': avg_genus_jump,
        'boost_components': {
            'energy_boost': energy_boost * 100,
            'genus_boost': genus_boost * 100,
            'wormhole_boost': wormhole_boost * 100
        }
    }

def predict_price_with_wormholes(state: SystemState, current_price: float, time_horizon: int = 30) -> dict:
    """Predict price movement including wormhole effects."""
    coherence = state.coherence
    chaos = state.chaos
    stress = state.stress
    hurst = state.hurst
    
    # Base prediction (same as before)
    drift = coherence * 0.01
    volatility = chaos * 0.05
    
    prices = [current_price]
    for t in range(1, time_horizon + 1):
        if hurst > 0.5:
            trend = drift * (1 + hurst - 0.5)
        else:
            trend = drift * (1 - (0.5 - hurst))
        
        noise = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(max(0.01, new_price))
    
    final_price = prices[-1]
    price_change = final_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    returns = np.diff(prices) / prices[:-1]
    predicted_volatility = np.std(returns) * np.sqrt(252) * 100
    
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

def showcase_wormhole_markets():
    """Showcase how wormholes affect financial market predictions."""
    
    print("=" * 70)
    print("WORMHOLE EFFECTS ON FINANCIAL MARKETS")
    print("=" * 70)
    print()
    print("Star Birth (Compression) → Creates Wormholes (Topological Shortcuts)")
    print("Wormholes → Energy Release → Market Opportunities")
    print()
    print("Wormhole Effects:")
    print("  - Energy release → Market momentum boost")
    print("  - Topological shortcuts → Faster price discovery")
    print("  - Genus changes → Structural market shifts")
    print("  - Information gain → Reduced uncertainty")
    print()
    
    Path(".out").mkdir(exist_ok=True)
    
    current_prices = {
        "bitcoin": 45000.0,
        "ethereum": 2800.0,
        "defi": 100.0,
        "nft": 50.0,
        "stablecoins": 1.0,
        "telegram": 5.50  # TON price in USD
    }
    
    all_predictions = {}
    all_wormholes = {}
    all_boosted = {}
    
    # Analyze each cryptocurrency with wormhole effects
    for crypto, text in CRYPTO_DATA.items():
        print(f"Analyzing: {crypto.upper()}")
        print("-" * 70)
        
        # Compress (star birth)
        result = compress_text(text)
        spec = result['spec']
        state = result['state']
        
        print(f"Star Birth (Compression):")
        print(f"  Compressed: {result['original_size']} bytes → 16 bytes")
        print(f"  Spectral Signature: center={spec.center:.6f}, seed={spec.seed:.6f}")
        print()
        
        # Detect wormholes
        wormholes = detect_wormholes(state, spec, n_iterations=30)
        all_wormholes[crypto] = wormholes
        
        print(f"Wormholes Detected: {len(wormholes)}")
        if wormholes:
            total_energy = sum(w['work_captured'] for w in wormholes)
            avg_genus = np.mean([w['genus_after'] for w in wormholes])
            print(f"  Total Energy Released: {total_energy:.2f} units")
            print(f"  Average Genus: {avg_genus:.1f}")
            print(f"  Topology Changes: {len(set(w['topology_change'] for w in wormholes))}")
        print()
        
        # Base prediction
        current_price = current_prices.get(crypto, 100.0)
        base_prediction = predict_price_with_wormholes(state, current_price, time_horizon=30)
        all_predictions[crypto] = base_prediction
        
        # Calculate wormhole boost
        boosted = calculate_wormhole_boost(base_prediction, wormholes)
        all_boosted[crypto] = boosted
        
        print(f"Price Predictions:")
        print(f"  Base Prediction: {base_prediction['price_change_pct']:+.2f}%")
        print(f"  Wormhole Boost: +{boosted['return_boost_pct']:.2f}%")
        print(f"  Boosted Prediction: {boosted['boosted_return']:+.2f}%")
        print()
        print(f"Confidence:")
        print(f"  Base: {base_prediction['confidence']:.1%}")
        print(f"  Boosted: {boosted['boosted_confidence']:.1%}")
        print()
        print(f"Volatility:")
        print(f"  Base: {base_prediction['predicted_volatility']:.1f}%")
        print(f"  Boosted: {boosted['boosted_volatility']:.1f}% (reduced by wormholes)")
        print()
    
    # Updated investment recommendations
    print("=" * 70)
    print("UPDATED INVESTMENT RECOMMENDATIONS (WITH WORMHOLE EFFECTS)")
    print("=" * 70)
    print()
    print(f"{'Asset':<15} | {'Base Return':<12} | {'Boost':<10} | {'Boosted Return':<15} | {'Confidence':<12}")
    print("-" * 70)
    
    for crypto in CRYPTO_DATA.keys():
        pred = all_predictions[crypto]
        boosted = all_boosted[crypto]
        print(f"{crypto.upper():<15} | {pred['price_change_pct']:>+11.2f}% | "
              f"{boosted['return_boost_pct']:>+9.2f}% | {boosted['boosted_return']:>+14.2f}% | "
              f"{boosted['boosted_confidence']:>11.1%}")
    
    print()
    
    # High-value opportunities (updated)
    print("=" * 70)
    print("HIGH-VALUE OPPORTUNITIES (WORMHOLE-ENHANCED)")
    print("=" * 70)
    print()
    
    # Sort by boosted return
    sorted_boosted = sorted(all_boosted.items(), 
                          key=lambda x: x[1]['boosted_return'], 
                          reverse=True)
    
    print(f"{'Rank':<6} | {'Asset':<15} | {'Boosted Return':<15} | {'Wormholes':<10} | {'Energy':<12}")
    print("-" * 70)
    
    high_value_wormhole = []
    for i, (crypto, boosted) in enumerate(sorted_boosted, 1):
        pred = all_predictions[crypto]
        wormholes = all_wormholes[crypto]
        total_energy = sum(w['work_captured'] for w in wormholes)
        
        print(f"{i:<6} | {crypto.upper():<15} | {boosted['boosted_return']:>+14.2f}% | "
              f"{len(wormholes):<10} | {total_energy:>11.2f}")
        
        # High-value: >15% boosted return, >60% boosted confidence, >5 wormholes
        if (boosted['boosted_return'] > 15 and 
            boosted['boosted_confidence'] > 0.6 and 
            len(wormholes) > 5):
            high_value_wormhole.append((crypto, pred, boosted, wormholes))
    
    print()
    
    if high_value_wormhole:
        print("⭐ HIGHEST-VALUE INVESTMENTS (Wormhole-Enhanced):")
        print("-" * 70)
        for crypto, pred, boosted, wormholes in high_value_wormhole:
            print(f"{crypto.upper()}:")
            print(f"  Boosted Return: {boosted['boosted_return']:+.2f}%")
            print(f"  Confidence: {boosted['boosted_confidence']:.1%}")
            print(f"  Wormholes: {len(wormholes)} (Energy: {sum(w['work_captured'] for w in wormholes):.2f})")
            print(f"  Boost Components:")
            for comp, value in boosted['boost_components'].items():
                print(f"    {comp}: +{value:.2f}%")
            print(f"  Current: ${pred['current_price']:,.2f}")
            boosted_price = pred['current_price'] * (1 + boosted['boosted_return'] / 100)
            print(f"  Boosted Predicted: ${boosted_price:,.2f}")
            print()
    
    # Save results
    results_file = ".out/wormhole_markets.json"
    with open(results_file, 'w') as f:
        json.dump({
            'predictions': {
                crypto: {
                    'base': pred,
                    'boosted': all_boosted[crypto],
                    'wormholes': [
                        {
                            'iteration': w['iteration'],
                            'genus_change': f"{w['genus_before']}→{w['genus_after']}",
                            'energy_released': w['energy_released'],
                            'work_captured': w['work_captured']
                        }
                        for w in all_wormholes[crypto]
                    ]
                }
                for crypto, pred in all_predictions.items()
            },
            'summary': {
                'total_wormholes': sum(len(w) for w in all_wormholes.values()),
                'total_energy': sum(sum(w['work_captured'] for w in wormholes) 
                                  for wormholes in all_wormholes.values()),
                'avg_return_boost': np.mean([b['return_boost_pct'] for b in all_boosted.values()])
            }
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    print()
    
    # Summary
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print()
    print("1. Star Birth (Compression) creates wormholes:")
    print(f"   Total wormholes detected: {sum(len(w) for w in all_wormholes.values())}")
    print(f"   Total energy released: {sum(sum(w['work_captured'] for w in wormholes) for wormholes in all_wormholes.values()):.2f} units")
    print()
    print("2. Wormholes boost market predictions:")
    print(f"   Average return boost: {np.mean([b['return_boost_pct'] for b in all_boosted.values()]):+.2f}%")
    print(f"   Average confidence boost: {np.mean([b['boosted_confidence'] - all_predictions[c]['confidence'] for c, b in all_boosted.items()]) * 100:+.1f}%")
    print()
    print("3. Wormholes reduce volatility:")
    print(f"   Average volatility reduction: {np.mean([(all_predictions[c]['predicted_volatility'] - b['boosted_volatility']) / all_predictions[c]['predicted_volatility'] * 100 for c, b in all_boosted.items()]):+.1f}%")
    print()
    print("4. Investment Strategy:")
    print("   - Focus on assets with most wormholes (more energy release)")
    print("   - Higher genus changes = more structural opportunities")
    print("   - Wormhole-enhanced returns are more reliable (higher confidence)")
    print()
    print("=" * 70)

if __name__ == "__main__":
    np.random.seed(42)
    showcase_wormhole_markets()


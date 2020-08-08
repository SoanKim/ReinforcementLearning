# Class for actor network
# Class for critic network
# Class for action noise
# Class for replay buffer
# Class for agent functionality
    # Memory, actor/critic networks, target nets, tau

# Initialize actor & critic networks 𝛍(s|𝛉^𝛍) Q(s, a|𝛉^Q)

# Initialize target networks with weights from main networks

# Initialize replay buffer R

# For a large number of episodes
    # Initialize noise process N for action exploration
    # Reset environment
    # For each step of game
        # at = 𝛍(st|𝛉^𝛍) + Nt
        # Take action at and get new state, reward
        # Store (st, at, rt, St+1, donet) in R
        # Sample random minibatch of (si, ai, ri, si+1) from R
        # yi = ri + rQ'(si+1, 𝛍'(si+1|𝛉^𝛍')|𝛉^Q)
        # Update critic by minimizing :=1/N𝚺(yi-Q(si, ai|𝛉^Q))^2
        # Update actor 𝛁𝛉≈1/N𝚺𝛁𝛉Q(s, a|𝛉^Q)|s=s, a=𝛍(st|𝛉^𝛍)
        # 𝛉^Q <- 𝛕𝛉^Q+(1-𝛄)𝛉^Q
        # 𝛉^𝛍 <- 𝛕𝛉^𝛍 + (1-𝛄)𝛉^𝛉^𝛍
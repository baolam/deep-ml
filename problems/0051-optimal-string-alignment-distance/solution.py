def OSA(source: str, target: str) -> int:
	# Your code here
	len_s = len(source)
	len_t = len(target)

	dp = [[0] * (len_t + 1) for _ in range(len_s + 1)]
	for i in range(len_s + 1):
		dp[i][0] = i
	for j in range(len_t + 1):
		dp[0][j] = j
	
	for i in range(1, len_s + 1):
		for j in range(1, len_t + 1):
			cost = 0 if source[i - 1] == target[j - 1] else 1

			dp[i][j] = min(
				dp[i - 1][j] + 1,
				dp[i][j - 1] + 1,
				dp[i-1][j-1] + cost
			)

			if i <= 1 or j <= 1:
				continue
			
			if source[i - 1] == target[j - 2] and source[i - 2] == target[j - 1]:
				dp[i][j] = min(dp[i][j], dp[i-2][j-2] + 1)
	
	return dp[len_s][len_t]
def findPasswordStrength(password: str) -> int:
    val = 1
    count = 0
    while val <= len(password):
      for index in range(len(password) + 1 - val):
        if index + val > len(password) - 1:
          substring = password[index:]
        else:
          substring = password[index:index+val]
        count += len(set(substring))
      val += 1

    print(count)
    return count

findPasswordStrength('')

import rsa

pubkey, privkey = rsa.newkeys(3072)
  
str1 = "I am okay"

print(type(pubkey))
print(str(pubkey))
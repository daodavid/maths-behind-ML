no_vaccinated_death = 158/100*95
vaccinated_death = 158-no_vaccinated_death
population= 6865696
vaccinated_population = (28.8/100)*6865696
unvaccinate_population = population - vaccinated_population
print(unvaccinate_population)
#vaccinated_population = 9916 - unvaccinate_population
print(vaccinated_population)
c = no_vaccinated_death / unvaccinate_population
print(c)
k =vaccinated_death/vaccinated_population
print(k)

print(c/k)


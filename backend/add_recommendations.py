# backend/add_recommendations.py
import asyncio
from app.core.neo4j_client import get_neo4j_driver

async def add_recs():
    driver = await get_neo4j_driver()
    if not driver:
        print('❌ Neo4j indisponible')
        return
    
    async with driver.session() as session:
        recs = {
            0: 'Examens complementaires necessaires',
            1: "Pas d'anomalie, suivi normal",
            2: 'Lesion benigne, suivi standard',
            3: 'Surveillance a 6 mois recommandee',
            4: 'Biopsie recommandee',
            5: 'Biopsie urgente recommandee',
            6: 'Malignite confirmee, traitement necessaire'
        }
        
        for level, rec in recs.items():
            try:
                await session.run(
                    'MATCH (b:BI_RADS {level: $level}) SET b.recommendation = $rec',
                    level=level, rec=rec
                )
                print(f'✅ BI-RADS {level} mis à jour')
            except Exception as e:
                print(f'❌ BI-RADS {level}: {e}')
        
        print('\n📋 Vérification:')
        result = await session.run('MATCH (b:BI_RADS) RETURN b.level AS level, b.recommendation AS rec ORDER BY b.level')
        async for record in result:
            print(f'  BI-RADS {record["level"]}: {record["rec"]}')

if __name__ == "__main__":
    asyncio.run(add_recs())
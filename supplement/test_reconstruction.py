import json,unittest
from pathlib import Path
from supplement.reconstruction import pool_summary,reconstruct
from reward import stage_reward

class SupplementTests(unittest.TestCase):
    def test_first_tie_uses_public_reward_and_random_mean_is_separate(self):
        rows=[dict(public_reward=1,hidden_reward=0,success=False),dict(public_reward=1,hidden_reward=1,success=True)]
        result=pool_summary(rows)
        self.assertEqual(result['selected_index'],0)
        self.assertEqual(result['random_hidden'],.5)
    def test_all_published_measurements_reconstruct(self):
        data=json.loads(Path(__file__).with_name('data.json').read_text(encoding='utf-8'))
        self.assertEqual(reconstruct(data,stage_reward),data['reconstruction'])

if __name__=='__main__':unittest.main()

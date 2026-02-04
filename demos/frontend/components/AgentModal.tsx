'use client';

import { useEffect } from 'react';
import { useCDSSStore, useModal } from '@/store/cdssStore';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';

export default function AgentModal() {
  const modal = useModal();
  const closeModal = useCDSSStore((state) => state.closeModal);

  // ESC key handler
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && modal.isOpen) {
        closeModal();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [modal.isOpen, closeModal]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (modal.isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [modal.isOpen]);

  return (
    <Dialog open={modal.isOpen} onOpenChange={closeModal}>
      <DialogContent className="max-w-4xl max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-lg">{modal.agentId} - Full Output</DialogTitle>
        </DialogHeader>
        <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
          <pre className="text-base leading-loose font-medium text-foreground whitespace-pre-wrap break-words">
            {modal.content}
          </pre>
        </div>
      </DialogContent>
    </Dialog>
  );
}

